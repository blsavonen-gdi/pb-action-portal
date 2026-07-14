"""
flow_network_interactive.py — pyvis-based interactive (draggable) flow network.

Mirrors visualizations.flow_network.build_flow_network but renders as an
HTML/JS graph where the user can grab and drag individual country nodes.

Public API
----------
build_flow_network_html(baci_df, active_years, countries, categories,
    min_flow, show_imports, show_exports, focal_country, layout,
    physics_enabled) -> str  (a self-contained HTML string)
"""

from __future__ import annotations

import math
from collections import defaultdict

import pandas as pd

from visualizations.flow_network import (
    _ABBREV,
    _ADV_HS_CAT,
    _CAT_COLORS,
    _CAT_DISPLAY,
    _country_positions,
    _grouping_maps,
    _label,
)


def _bubble_sizes(
    countries: list[str],
    baci_df: pd.DataFrame,
    active_years: list[int],
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    subset = baci_df[baci_df["Year"].isin(active_years)]
    n_years = max(len(active_years), 1)
    imp_vols: dict[str, float] = {}
    exp_vols: dict[str, float] = {}
    for c in countries:
        imp_vols[c] = float(subset[subset["Importer"] == c]["actual_lead"].sum()) / n_years
        exp_vols[c] = float(subset[subset["Exporter"] == c]["actual_lead"].sum()) / n_years
    total_vols = {c: imp_vols[c] + exp_vols[c] for c in countries}
    sqrt_vals = {c: math.sqrt(v) for c, v in total_vols.items()}
    min_s = min(sqrt_vals.values()) if sqrt_vals else 0.0
    max_s = max(sqrt_vals.values()) if sqrt_vals else 1.0
    sizes: dict[str, float] = {}
    for c, s in sqrt_vals.items():
        t = (s - min_s) / (max_s - min_s) if max_s > min_s else 0.5
        # vis.js radii: 12 – 45 px
        sizes[c] = 12.0 + t * 33.0
    return sizes, imp_vols, exp_vols


def build_flow_network_html(
    baci_df: pd.DataFrame,
    active_years: list[int],
    countries: list[str],
    categories: list[str],
    min_flow: float,
    show_imports: bool,
    show_exports: bool,
    focal_country: str | None = None,
    layout: str = "circle",
    physics_enabled: bool = False,
    height_px: int = 650,
    focal_only: bool = False,
    hidden_pairs: set[frozenset[str]] | None = None,
    prune_isolated: bool = False,
    grouping: str = "trade",
    products: tuple[int, ...] | None = None,
) -> str:
    """Return a self-contained HTML string for an interactive draggable flow network.

    physics_enabled=False means nodes stay where they are placed / dragged.
    physics_enabled=True runs a live force simulation.
    """
    try:
        from pyvis.network import Network
    except ImportError as e:
        raise ImportError(
            "pyvis is required for the interactive Flow Network view. "
            "Install with: pip install pyvis"
        ) from e

    df = baci_df[baci_df["Year"].isin(active_years)]
    cat_colors, cat_display = _grouping_maps(grouping)
    if grouping == "advanced":
        df = df.assign(category=df["Product"].map(_ADV_HS_CAT))
    if products is not None:
        df = df[df["Product"].isin(products)]
    n_years = max(len(active_years), 1)
    country_set = set(countries)

    flows = (
        df[
            df["Exporter"].isin(country_set)
            & df["Importer"].isin(country_set)
            & df["category"].isin(categories)
            & (df["Exporter"] != df["Importer"])
        ]
        .groupby(["Exporter", "Importer", "category"])["actual_lead"]
        .sum()
        .reset_index()
    )
    flows["actual_lead"] = flows["actual_lead"] / n_years
    flows = flows[flows["actual_lead"] >= min_flow]

    # Direction filter — relative to the focal country. Every arrow between two
    # selected countries is simultaneously an import (for its Importer) and an
    # export (for its Exporter), so "imports vs exports" is only well-defined
    # relative to a chosen focal country. Without one, both flags are True
    # ("Both") and nothing is filtered.
    if focal_country and not (show_imports and show_exports):
        if show_imports and not show_exports:
            flows = flows[flows["Importer"] == focal_country]
        elif show_exports and not show_imports:
            flows = flows[flows["Exporter"] == focal_country]
        else:  # neither direction selected → show nothing
            flows = flows.iloc[0:0]

    if focal_only and focal_country:
        flows = flows[
            (flows["Exporter"] == focal_country)
            | (flows["Importer"] == focal_country)
        ]

    if hidden_pairs and not flows.empty:
        _hidden_key = flows.apply(
            lambda r: frozenset([r["Exporter"], r["Importer"]]),
            axis=1,
        )
        flows = flows[~_hidden_key.isin(hidden_pairs)]

    if prune_isolated and focal_country and focal_country in country_set:
        connected = (
            set(flows.loc[flows["Exporter"] == focal_country, "Importer"])
            | set(flows.loc[flows["Importer"] == focal_country, "Exporter"])
        )
        connected.add(focal_country)
        countries = [c for c in countries if c in connected]
        country_set = set(countries)
        flows = flows[
            flows["Exporter"].isin(country_set) & flows["Importer"].isin(country_set)
        ]

    # ── Initial positions ─────────────────────────────────────────────────────
    positions = _country_positions(
        countries,
        focal_country=focal_country,
        layout=layout,
        baci_df=df,
        active_years=active_years,
        categories=categories,
    )
    sizes, imp_vols, exp_vols = _bubble_sizes(countries, baci_df, active_years)
    max_flow = flows["actual_lead"].max() if not flows.empty else 1.0

    # ── Build the network ─────────────────────────────────────────────────────
    net = Network(
        height=f"{height_px}px",
        width="100%",
        bgcolor="#f8f9fa",
        font_color="#111111",
        directed=True,
        notebook=False,
        cdn_resources="in_line",
    )
    net.toggle_physics(physics_enabled)

    # Map the 0-100 schematic coordinates to pixel space that vis.js will use.
    # (vis.js y-axis points down like our schematic, so no flip needed.)
    scale = 12.0  # 1 unit on 0-100 canvas → 12 px
    origin_x, origin_y = -50 * scale, -50 * scale

    for c in countries:
        x, y = positions[c]
        px, py = origin_x + x * scale, origin_y + y * scale
        total_trade = imp_vols[c] + exp_vols[c]
        title = (
            f"{c}\n"
            f"Total imports: {imp_vols[c]:,.0f} t Pb\n"
            f"Total exports: {exp_vols[c]:,.0f} t Pb\n"
            f"Total trade volume: {total_trade:,.0f} t Pb"
        )
        net.add_node(
            c,
            label=_label(c),
            title=title,
            x=px,
            y=py,
            size=sizes[c],
            color={
                "background": "#37474F",
                "border": "#ffffff",
                "highlight": {"background": "#546E7A", "border": "#000000"},
            },
            font={"size": 14, "color": "#111111"},
            shape="dot",
            physics=physics_enabled,
        )

    # Pin the focal country in place even when physics is on
    if focal_country and focal_country in countries:
        # vis.js supports `fixed: true` per node
        for node in net.nodes:
            if node["id"] == focal_country:
                node["fixed"] = {"x": True, "y": True}

    for _, row in flows.iterrows():
        exp, imp, cat = row["Exporter"], row["Importer"], row["category"]
        vol = float(row["actual_lead"])
        color = cat_colors.get(cat, "#555")
        width = max(1.5, 1.0 + math.sqrt(vol / max_flow) * 7.0)
        title = f"{exp} → {imp}\n{cat_display.get(cat, cat)}: {vol:,.0f} t Pb"
        net.add_edge(
            exp,
            imp,
            value=width,
            width=width,
            color=color,
            title=title,
            arrows="to",
            smooth={"enabled": True, "type": "curvedCW", "roundness": 0.15},
        )

    # ── vis.js options ────────────────────────────────────────────────────────
    # A moderate barnesHut simulation, but disabled unless the user turns it on.
    options = """
    {
      "physics": {
        "enabled": %s,
        "barnesHut": {
          "gravitationalConstant": -8000,
          "centralGravity": 0.15,
          "springLength": 140,
          "springConstant": 0.03,
          "damping": 0.35,
          "avoidOverlap": 0.5
        },
        "stabilization": {"iterations": 150},
        "minVelocity": 0.5
      },
      "interaction": {
        "dragNodes": true,
        "dragView": true,
        "zoomView": true,
        "hover": true,
        "tooltipDelay": 120
      },
      "edges": {
        "smooth": {"enabled": true, "type": "curvedCW", "roundness": 0.15}
      },
      "nodes": {
        "borderWidth": 1.5,
        "borderWidthSelected": 2.5
      }
    }
    """ % ("true" if physics_enabled else "false")
    net.set_options(options)

    html = net.generate_html(notebook=False)

    # Ensure the graph fills its Streamlit iframe cleanly.
    html = html.replace(
        '<div id="mynetwork" class="card-body">',
        f'<div id="mynetwork" class="card-body" style="height: {height_px}px;">',
    )
    return html
