"""
trade_map.py — Plotly choropleth maps for BACI bilateral trade flows.

Two entry points:
  build_total_volume_map  — all countries shaded by total trade volume
  build_bilateral_map     — selected country's partners colored by flow direction
"""

import numpy as np
import plotly.graph_objects as go
import pandas as pd

from model.data_loader import BACI_TO_STANDARD_NAME


def _std(name: str) -> str | None:
    """Map a BACI country name to a Plotly-recognised standard name.
    Returns None for territories that cannot be rendered on the choropleth."""
    return BACI_TO_STANDARD_NAME.get(name, name)  # passthrough if not in dict


def _base_geo() -> dict:
    return dict(
        showframe=False,
        showcoastlines=True,
        coastlinecolor="rgb(160,160,160)",
        projection_type="natural earth",
        showland=True,
        landcolor="rgb(235,235,235)",
        showocean=True,
        oceancolor="rgb(210,225,240)",
        showlakes=False,
        showcountries=True,
        countrycolor="rgb(200,200,200)",
        countrywidth=0.5,
    )


# ── Total volume map ──────────────────────────────────────────────────────────

_NET_COLORSCALE = [
    [0.00, "rgb(165,0,38)"],     # deep red   → large net importer
    [0.20, "rgb(244,109,67)"],   # orange-red
    [0.40, "rgb(253,204,138)"],  # pale orange
    [0.50, "rgb(255,255,255)"],  # white      → balanced
    [0.60, "rgb(166,217,106)"],  # pale green
    [0.80, "rgb(102,189,99)"],   # medium green
    [1.00, "rgb(0,104,55)"],     # deep green → large net exporter
]


def build_total_volume_map(
    baci_df: pd.DataFrame,
    year: int,
    category: str | list[str],
    category_label: str | None = None,
) -> go.Figure:
    """
    Choropleth shading every country by its net trade position
    (exports minus imports) for the given year and product category.

    Color scale: deep red (large net importer) → white (balanced) → deep green (large net exporter).
    Z-axis uses a signed-log transform: sign(net) * log1p(|net|), normalised to [-1, 1].
    """
    cats = [category] if isinstance(category, str) else category
    df = baci_df[
        (baci_df["Year"] == year) & (baci_df["category"].isin(cats))
    ]

    cat_display = category_label or category
    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            title=dict(text=f"No {cat_display} trade recorded for {year}", x=0),
            height=500,
            geo=_base_geo(),
            margin=dict(l=0, r=0, t=40, b=0),
        )
        return fig

    exp = df.groupby("Exporter")["actual_lead"].sum().rename("exp_t")
    imp = df.groupby("Importer")["actual_lead"].sum().rename("imp_t")
    total = pd.concat([exp, imp], axis=1).fillna(0)
    total["total_t"] = total["exp_t"] + total["imp_t"]
    total = total[total["total_t"] > 0].reset_index().rename(columns={"index": "country"})

    total["std_name"] = total["country"].map(_std)
    total = total[total["std_name"].notna()].copy()

    total["net_t"] = total["exp_t"] - total["imp_t"]
    total["signed_log"] = np.sign(total["net_t"]) * np.log1p(np.abs(total["net_t"]))
    max_abs = total["signed_log"].abs().max()
    total["z"] = (total["signed_log"] / max_abs).clip(-1.0, 1.0) if max_abs > 0 else 0.0

    def _net_label(net: float) -> str:
        if net > 100:
            return f"Net exporter (+{net:,.0f} t Pb)"
        elif net < -100:
            return f"Net importer ({net:,.0f} t Pb)"
        return "Roughly balanced"

    total["hover"] = total.apply(
        lambda r: (
            f"<b>{r['country']}</b><br>"
            f"Exports: {r['exp_t']:,.0f} t Pb<br>"
            f"Imports: {r['imp_t']:,.0f} t Pb<br>"
            f"Net: {r['net_t']:+,.0f} t Pb  ({_net_label(r['net_t'])})"
        ),
        axis=1,
    )

    fig = go.Figure(
        go.Choropleth(
            locations=total["std_name"],
            z=total["z"],
            locationmode="country names",
            colorscale=_NET_COLORSCALE,
            zmin=-1,
            zmax=1,
            colorbar=dict(
                title=dict(text="Net position", side="right"),
                tickvals=[-1, -0.5, 0, 0.5, 1],
                ticktext=["Net Importer", "↑", "Balanced", "↑", "Net Exporter"],
                len=0.65,
                thickness=14,
            ),
            hovertext=total["hover"],
            hovertemplate="%{hovertext}<extra></extra>",
            marker=dict(line=dict(color="white", width=0.5)),
        )
    )

    fig.update_layout(
        title=dict(
            text=f"<b>{cat_display} trade — {year}</b>  (net position: green = net exporter, red = net importer)",
            x=0,
            font=dict(size=14),
        ),
        height=500,
        geo=_base_geo(),
        margin=dict(l=0, r=0, t=45, b=0),
    )
    return fig


# ── Bilateral map ─────────────────────────────────────────────────────────────

_BILATERAL_COLORSCALE = [
    [0.00, "rgb(165,0,38)"],     # deep red  → pure exporter
    [0.20, "rgb(244,109,67)"],   # orange-red
    [0.40, "rgb(253,204,138)"],  # pale orange
    [0.50, "rgb(255,255,255)"],  # white     → balanced
    [0.60, "rgb(166,217,106)"],  # pale green
    [0.80, "rgb(102,189,99)"],   # medium green
    [1.00, "rgb(0,104,55)"],     # deep green → pure importer
]


def build_bilateral_map(
    baci_df: pd.DataFrame,
    year: int,
    category: str | list[str],
    selected_country: str,
    category_label: str | None = None,
) -> go.Figure:
    """
    Choropleth showing trade partners of `selected_country`.

    Color encodes direction and balance (from selected country's perspective):
      Deep green: partner sends product TO selected country (selected is net importer)
      Deep red:   partner receives FROM selected country (selected is net exporter)
      White:      balanced bilateral flow
    Intensity is scaled by log-normalized bilateral volume so small flows are pale.
    """
    cats = [category] if isinstance(category, str) else category
    df = baci_df[
        (baci_df["Year"] == year) & (baci_df["category"].isin(cats))
    ]

    # Aggregate exports and imports for the selected country
    exp_df = (
        df[df["Exporter"] == selected_country]
        .groupby("Importer")["actual_lead"].sum()
        .rename("EXP_to_partner")
    )
    imp_df = (
        df[df["Importer"] == selected_country]
        .groupby("Exporter")["actual_lead"].sum()
        .rename("IMP_from_partner")
    )

    partners = pd.concat([exp_df, imp_df], axis=1).fillna(0)
    partners["total_bilateral"] = partners["EXP_to_partner"] + partners["IMP_from_partner"]
    partners = partners[partners["total_bilateral"] > 0].copy()

    fig = go.Figure()

    sel_std = _std(selected_country)

    if partners.empty:
        _apply_bilateral_layout(fig, selected_country, year, category, category_label)
        _add_selected_highlight(fig, sel_std, selected_country)
        return fig

    partners["net"] = partners["IMP_from_partner"] - partners["EXP_to_partner"]
    partners["balance_ratio"] = partners["net"] / partners["total_bilateral"]

    # Volume-based intensity: log-normalize so small flows are pale (near 0)
    max_vol = partners["total_bilateral"].max()
    partners["log_intensity"] = (
        np.log1p(partners["total_bilateral"]) / np.log1p(max_vol)
    ).clip(0.0, 1.0)

    # Composite z: direction × intensity.
    # balance_ratio=+1 (pure importer) × high intensity → z near +1 → vivid green
    # balance_ratio=-1 (pure exporter) × high intensity → z near -1 → vivid red
    # small volume → log_intensity≈0 → z≈0 → pale yellow regardless of direction
    partners["z"] = partners["balance_ratio"] * partners["log_intensity"]

    partners.index.name = "partner"
    partners = partners.reset_index()
    partners["std_name"] = partners["partner"].map(_std)
    partners = partners[partners["std_name"].notna()].copy()

    # Hover tooltips
    def _dir(net: float) -> str:
        if net > 50:
            return f"net flow to {selected_country}"
        elif net < -50:
            return f"net flow from {selected_country}"
        return "balanced"

    # From the *partner's* perspective:
    # "Imports from [selected]" = what partner imports from selected = selected's EXP_to_partner
    # "Exports to [selected]"   = what partner exports to selected   = selected's IMP_from_partner
    partners["hover"] = partners.apply(
        lambda r: (
            f"<b>{r['partner']}</b><br>"
            f"Imports from {selected_country}: {r['EXP_to_partner']:,.0f} t Pb<br>"
            f"Exports to {selected_country}: {r['IMP_from_partner']:,.0f} t Pb<br>"
            f"Net: {abs(r['net']):,.0f} t Pb ({_dir(r['net'])})"
        ),
        axis=1,
    )

    fig.add_trace(
        go.Choropleth(
            locations=partners["std_name"],
            z=partners["z"],
            locationmode="country names",
            colorscale=_BILATERAL_COLORSCALE,
            zmin=-1,
            zmax=1,
            colorbar=dict(
                title=dict(text="Flow direction", side="right"),
                tickvals=[-1, -0.5, 0, 0.5, 1],
                ticktext=[f"{selected_country} Exports To", "↑", "Balanced", "↑", f"{selected_country} Imports From"],
                len=0.65,
                thickness=14,
            ),
            hovertext=partners["hover"],
            hovertemplate="%{hovertext}<extra></extra>",
            marker=dict(line=dict(color="white", width=0.5)),
        )
    )

    _add_selected_highlight(fig, sel_std, selected_country)
    _apply_bilateral_layout(fig, selected_country, year, category, category_label)
    return fig


def _add_selected_highlight(
    fig: go.Figure, std_name: str | None, display_name: str
) -> None:
    """Add the selected country as a separate trace with a black border and bright blue fill."""
    if std_name is None:
        return
    fig.add_trace(
        go.Choropleth(
            locations=[std_name],
            z=[0],
            locationmode="country names",
            showscale=False,
            colorscale=[[0, "#1565C0"], [1, "#1565C0"]],
            zmin=-1,
            zmax=1,
            hovertemplate=f"<b>{display_name}</b> (selected)<extra></extra>",
            marker=dict(line=dict(color="black", width=2.5)),
        )
    )


def _apply_bilateral_layout(
    fig: go.Figure, country: str, year: int, category: str,
    category_label: str | None = None,
) -> None:
    cat_display = category_label or category
    fig.update_layout(
        title=dict(
            text=f"<b>{country} — {cat_display} trade partners, {year}</b>",
            x=0,
            font=dict(size=14),
        ),
        height=500,
        geo=_base_geo(),
        margin=dict(l=0, r=0, t=45, b=0),
    )


# ── Bilateral trade balance map (ratio-based, volume-independent) ─────────────

_BALANCE_COLORSCALE = [
    [0.00, "rgb(127,0,212)"],    # deep purple  → dominant receiver
    [0.20, "rgb(180,100,240)"],  # medium purple
    [0.40, "rgb(220,190,255)"],  # light purple
    [0.50, "rgb(255,255,255)"],  # white → balanced
    [0.60, "rgb(255,210,150)"],  # light orange
    [0.80, "rgb(255,140,50)"],   # medium orange
    [1.00, "rgb(200,80,0)"],     # deep orange  → dominant sender
]


def build_bilateral_balance_map(
    baci_df: pd.DataFrame,
    year: int,
    category: str | list[str],
    selected_country: str,
    category_label: str | None = None,
) -> go.Figure:
    """
    Choropleth coloring each country by its TRADE BALANCE RATIO with selected_country.

    ratio = (exports_to_selected - imports_from_selected) /
            (exports_to_selected + imports_from_selected)

    ratio = +1 → partner sends everything, receives nothing → deep orange
    ratio =  0 → balanced → white
    ratio = -1 → partner receives everything, sends nothing → deep purple

    This is volume-independent: two countries with the same ratio get the same
    color regardless of absolute trade volume.
    Countries with zero bilateral trade are not shown (gray land color).
    """
    cats = [category] if isinstance(category, str) else category
    df = baci_df[
        (baci_df["Year"] == year) & (baci_df["category"].isin(cats))
    ]
    cat_display = category_label or (cats[0] if len(cats) == 1 else "All products")

    exp_df = (
        df[df["Exporter"] == selected_country]
        .groupby("Importer")["actual_lead"].sum()
        .rename("EXP_to_partner")
    )
    imp_df = (
        df[df["Importer"] == selected_country]
        .groupby("Exporter")["actual_lead"].sum()
        .rename("IMP_from_partner")
    )

    partners = pd.concat([exp_df, imp_df], axis=1).fillna(0)
    partners["total_bilateral"] = partners["EXP_to_partner"] + partners["IMP_from_partner"]
    partners = partners[partners["total_bilateral"] > 0].copy()

    fig = go.Figure()
    sel_std = _std(selected_country)

    if not partners.empty:
        # From partner's perspective: sends = IMP_from_partner, receives = EXP_to_partner
        partners["sends"]    = partners["IMP_from_partner"]
        partners["receives"] = partners["EXP_to_partner"]
        partners["ratio"] = (partners["sends"] - partners["receives"]) / partners["total_bilateral"]

        partners.index.name = "partner"
        partners = partners.reset_index()
        partners["std_name"] = partners["partner"].map(_std)
        partners = partners[partners["std_name"].notna()].copy()

        def _dir_label(ratio: float) -> str:
            if ratio > 0.1:
                return f"net sender to {selected_country}"
            elif ratio < -0.1:
                return f"net receiver from {selected_country}"
            return "roughly balanced"

        partners["hover"] = partners.apply(
            lambda r: (
                f"<b>{r['partner']}</b><br>"
                f"Sends to {selected_country}: {r['sends']:,.0f} t Pb<br>"
                f"Receives from {selected_country}: {r['receives']:,.0f} t Pb<br>"
                f"Balance ratio: {r['ratio']:+.2f} ({_dir_label(r['ratio'])})<br>"
                f"<i>Color shows balance ratio, not volume</i>"
            ),
            axis=1,
        )

        fig.add_trace(
            go.Choropleth(
                locations=partners["std_name"],
                z=partners["ratio"],
                locationmode="country names",
                colorscale=_BALANCE_COLORSCALE,
                zmin=-1,
                zmax=1,
                colorbar=dict(
                    title=dict(text="Balance ratio", side="right"),
                    tickvals=[-1, -0.5, 0, 0.5, 1],
                    ticktext=["Pure receiver", "↑", "Balanced", "↑", "Pure sender"],
                    len=0.65,
                    thickness=14,
                ),
                hovertext=partners["hover"],
                hovertemplate="%{hovertext}<extra></extra>",
                marker=dict(line=dict(color="white", width=0.5)),
            )
        )

    _add_selected_highlight(fig, sel_std, selected_country)
    _apply_bilateral_layout(fig, selected_country, year, cats[0] if len(cats) == 1 else "All products", cat_display)
    return fig


# ── Regional bilateral map ────────────────────────────────────────────────────

def build_region_bilateral_map(
    baci_df: pd.DataFrame,
    active_years: list[int],
    category: str | list[str],
    selected_region: str,
    region_map: dict[str, str],
    category_label: str | None = None,
) -> go.Figure:
    """
    Choropleth showing trade partners of an entire UN region, treating it as a
    single aggregated unit.

    Color encodes direction and balance (from the selected region's perspective):
      Deep green : partner sends product TO the region (region is net importer)
      Deep red   : partner receives product FROM the region (region is net exporter)
      White      : balanced bilateral flow
    Intensity scales with log-normalised bilateral volume.

    The selected region's own countries are highlighted in bright blue (#1565C0)
    with a black border.

    Hover per partner country shows:
      "[Country] ([Region])
       Imports from [selected region]: X,XXX t Pb
       Exports to [selected region]: Y,XXX t Pb
       Net: Z,XXX t Pb"
    """
    cats = [category] if isinstance(category, str) else category
    n_years = len(active_years)
    df = baci_df[
        baci_df["Year"].isin(active_years) & baci_df["category"].isin(cats)
    ]

    cat_display = category_label or (cats[0] if len(cats) == 1 else "All products")

    # Countries belonging to the selected region
    region_countries = {c for c, r in region_map.items() if r == selected_region}

    # Cross-region flows only (exclude intra-regional)
    cross = df[~(df["Exporter"].isin(region_countries) & df["Importer"].isin(region_countries))]

    # EXP from region to each external partner (annualised)
    exp_from_region = (
        cross[cross["Exporter"].isin(region_countries) & ~cross["Importer"].isin(region_countries)]
        .groupby("Importer")["actual_lead"].sum()
        / n_years
    ).rename("EXP_to_partner")

    # IMP to region from each external partner (annualised)
    imp_to_region = (
        cross[cross["Importer"].isin(region_countries) & ~cross["Exporter"].isin(region_countries)]
        .groupby("Exporter")["actual_lead"].sum()
        / n_years
    ).rename("IMP_from_partner")

    partners = pd.concat([exp_from_region, imp_to_region], axis=1).fillna(0)
    partners["total_bilateral"] = partners["EXP_to_partner"] + partners["IMP_from_partner"]
    partners = partners[partners["total_bilateral"] > 0].copy()

    fig = go.Figure()

    if not partners.empty:
        partners["net"] = partners["IMP_from_partner"] - partners["EXP_to_partner"]
        partners["balance_ratio"] = partners["net"] / partners["total_bilateral"]

        max_vol = partners["total_bilateral"].max()
        partners["log_intensity"] = (
            np.log1p(partners["total_bilateral"]) / np.log1p(max_vol)
        ).clip(0.0, 1.0)

        partners["z"] = partners["balance_ratio"] * partners["log_intensity"]

        partners.index.name = "partner"
        partners = partners.reset_index()
        partners["region"] = partners["partner"].map(
            lambda c: region_map.get(c, "Other")
        )
        partners["std_name"] = partners["partner"].map(_std)
        # Exclude region's own countries from the partner choropleth layer
        partners = partners[
            partners["std_name"].notna()
            & ~partners["partner"].isin(region_countries)
        ].copy()

        partners["hover"] = partners.apply(
            lambda r: (
                f"<b>{r['partner']}</b> ({r['region']})<br>"
                f"Imports from {selected_region}: {r['EXP_to_partner']:,.0f} t Pb<br>"
                f"Exports to {selected_region}: {r['IMP_from_partner']:,.0f} t Pb<br>"
                f"Net: {r['net']:,.0f} t Pb"
            ),
            axis=1,
        )

        fig.add_trace(
            go.Choropleth(
                locations=partners["std_name"],
                z=partners["z"],
                locationmode="country names",
                colorscale=_BILATERAL_COLORSCALE,
                zmin=-1,
                zmax=1,
                colorbar=dict(
                    title=dict(text="Flow direction", side="right"),
                    tickvals=[-1, -0.5, 0, 0.5, 1],
                    ticktext=[f"{selected_region} Exports To", "↑", "Balanced", "↑", f"{selected_region} Imports From"],
                    len=0.65,
                    thickness=14,
                ),
                hovertext=partners["hover"],
                hovertemplate="%{hovertext}<extra></extra>",
                marker=dict(line=dict(color="white", width=0.5)),
            )
        )

    _add_region_highlight(fig, region_countries, selected_region)
    _apply_region_layout(fig, selected_region, active_years, cat_display)
    return fig


def build_region_balance_map(
    baci_df: pd.DataFrame,
    active_years: list[int],
    category: str | list[str],
    selected_region: str,
    region_map: dict[str, str],
    category_label: str | None = None,
) -> go.Figure:
    """
    Regional analogue of build_bilateral_balance_map.

    Each external partner country is colored by its BALANCE RATIO with the
    selected region — volume-independent — using the same purple/orange scale
    as the Country view's "Bilateral trade balance" mode.

    ratio = (sends_to_region - receives_from_region) / total_bilateral
      +1 → partner sends everything, receives nothing → deep orange
       0 → balanced → white
      -1 → partner receives everything, sends nothing → deep purple

    The selected region's own countries are highlighted in bright blue.
    """
    cats = [category] if isinstance(category, str) else category
    n_years = max(len(active_years), 1)
    df = baci_df[
        baci_df["Year"].isin(active_years) & baci_df["category"].isin(cats)
    ]
    cat_display = category_label or (cats[0] if len(cats) == 1 else "All products")

    region_countries = {c for c, r in region_map.items() if r == selected_region}
    cross = df[~(df["Exporter"].isin(region_countries) & df["Importer"].isin(region_countries))]

    # From the region's perspective:
    #   EXP_to_partner  = region sends to partner  = partner receives from region
    #   IMP_from_partner = region receives from partner = partner sends to region
    exp_from_region = (
        cross[cross["Exporter"].isin(region_countries) & ~cross["Importer"].isin(region_countries)]
        .groupby("Importer")["actual_lead"].sum()
        / n_years
    ).rename("EXP_to_partner")

    imp_to_region = (
        cross[cross["Importer"].isin(region_countries) & ~cross["Exporter"].isin(region_countries)]
        .groupby("Exporter")["actual_lead"].sum()
        / n_years
    ).rename("IMP_from_partner")

    partners = pd.concat([exp_from_region, imp_to_region], axis=1).fillna(0)
    partners["total_bilateral"] = partners["EXP_to_partner"] + partners["IMP_from_partner"]
    partners = partners[partners["total_bilateral"] > 0].copy()

    fig = go.Figure()

    if not partners.empty:
        partners["sends"]    = partners["IMP_from_partner"]
        partners["receives"] = partners["EXP_to_partner"]
        partners["ratio"] = (partners["sends"] - partners["receives"]) / partners["total_bilateral"]

        partners.index.name = "partner"
        partners = partners.reset_index()
        partners["region"] = partners["partner"].map(lambda c: region_map.get(c, "Other"))
        partners["std_name"] = partners["partner"].map(_std)
        partners = partners[
            partners["std_name"].notna()
            & ~partners["partner"].isin(region_countries)
        ].copy()

        def _dir_label(ratio: float) -> str:
            if ratio > 0.1:
                return f"net sender to {selected_region}"
            elif ratio < -0.1:
                return f"net receiver from {selected_region}"
            return "roughly balanced"

        partners["hover"] = partners.apply(
            lambda r: (
                f"<b>{r['partner']}</b> ({r['region']})<br>"
                f"Sends to {selected_region}: {r['sends']:,.0f} t Pb<br>"
                f"Receives from {selected_region}: {r['receives']:,.0f} t Pb<br>"
                f"Balance ratio: {r['ratio']:+.2f} ({_dir_label(r['ratio'])})<br>"
                f"<i>Color shows balance ratio, not volume</i>"
            ),
            axis=1,
        )

        fig.add_trace(
            go.Choropleth(
                locations=partners["std_name"],
                z=partners["ratio"],
                locationmode="country names",
                colorscale=_BALANCE_COLORSCALE,
                zmin=-1,
                zmax=1,
                colorbar=dict(
                    title=dict(text="Balance ratio", side="right"),
                    tickvals=[-1, -0.5, 0, 0.5, 1],
                    ticktext=["Pure receiver", "↑", "Balanced", "↑", "Pure sender"],
                    len=0.65,
                    thickness=14,
                ),
                hovertext=partners["hover"],
                hovertemplate="%{hovertext}<extra></extra>",
                marker=dict(line=dict(color="white", width=0.5)),
            )
        )

    _add_region_highlight(fig, region_countries, selected_region)
    _apply_region_layout(fig, selected_region, active_years, cat_display)
    return fig


def _add_region_highlight(
    fig: go.Figure, region_countries: set[str], selected_region: str
) -> None:
    region_std_names = [_std(c) for c in region_countries if _std(c) is not None]
    if not region_std_names:
        return
    fig.add_trace(
        go.Choropleth(
            locations=region_std_names,
            z=[0] * len(region_std_names),
            locationmode="country names",
            showscale=False,
            colorscale=[[0, "#1565C0"], [1, "#1565C0"]],
            zmin=-1,
            zmax=1,
            hovertemplate=f"<b>%{{location}}</b> ({selected_region}) — Selected region<extra></extra>",
            marker=dict(line=dict(color="black", width=1.5)),
        )
    )


def _apply_region_layout(
    fig: go.Figure, selected_region: str, active_years: list[int], cat_display: str
) -> None:
    period_str = (
        f"{active_years[0]}–{active_years[-1]}" if len(active_years) > 1
        else str(active_years[0])
    )
    fig.update_layout(
        title=dict(
            text=f"<b>{selected_region} — {cat_display} trade partners, {period_str}</b>",
            x=0,
            font=dict(size=14),
        ),
        height=500,
        geo=_base_geo(),
        margin=dict(l=0, r=0, t=45, b=0),
    )
