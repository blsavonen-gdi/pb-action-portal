"""
trade_composition_map.py — Global lead trade composition map.

Colors each country by the mix of its imports or exports across three categories:
  Battery Inputs  (refined lead, oxides)  → deep Blue
  New Batteries                            → deep Yellow
  Battery Waste   (scrap, used batteries) → deep Red

Secondary mixes (via RYB circular-mean color model):
  Inputs + New Batteries → Green
  New Batteries + Waste  → Orange
  Inputs + Waste         → Purple
  Equal mix              → White

Methodology: each category's share is treated as a weight in RYB color space.
The three corner hues are placed 120° apart (equidistant) so the circular mean
collapses to zero (white) at equal shares.
"""

from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from visualizations.lead_accumulation import HS_META, load_master_baci

# ── Category → HS code mapping ────────────────────────────────────────────────
_CAT_HS: dict[str, list[int]] = {
    "Battery Inputs": [282410, 282490, 780110, 780191, 780199, 850790],
    "New Batteries":  [850710, 850720],
    "Battery Waste":  [780200, 854810, 854911],  # 854810=HS12, 854911=HS22
}
_HS_TO_CAT: dict[int, str] = {
    code: cat for cat, codes in _CAT_HS.items() for code in codes
}
_CATEGORIES = ["Battery Inputs", "New Batteries", "Battery Waste"]

# RYB hue angles (equidistant at 120°) — gives white at equal-weight center
_RYB_RAD = {
    "Battery Waste":  math.radians(0),    # Red
    "New Batteries":  math.radians(120),  # Yellow
    "Battery Inputs": math.radians(240),  # Blue
}


# ── Color math ────────────────────────────────────────────────────────────────

def _ryb_to_rgb_hue(ryb_deg: float) -> float:
    """Piecewise-linear RYB → RGB hue (degrees). Keeps red, yellow, blue as-is;
    maps RYB green (180°) to RGB green (120°) and compresses orange range."""
    cps = [(0, 0), (60, 30), (120, 60), (180, 120), (240, 240), (300, 300), (360, 360)]
    d = ryb_deg % 360
    for i in range(len(cps) - 1):
        r0, g0 = cps[i]
        r1, g1 = cps[i + 1]
        if r0 <= d <= r1:
            t = (d - r0) / (r1 - r0)
            return g0 + t * (g1 - g0)
    return d


def _hsl_to_hex(h: float, s: float, l: float) -> str:
    """HSL (h in [0,360], s/l in [0,1]) → #rrggbb string."""
    h /= 360.0
    if s < 1e-6:
        v = int(round(l * 255))
        return f"#{v:02x}{v:02x}{v:02x}"

    def _hue2rgb(p: float, q: float, t: float) -> float:
        t %= 1.0
        if t < 1 / 6:
            return p + (q - p) * 6 * t
        if t < 0.5:
            return q
        if t < 2 / 3:
            return p + (q - p) * (2 / 3 - t) * 6
        return p

    q = l * (1 + s) if l < 0.5 else l + s - l * s
    p = 2 * l - q
    r = int(round(_hue2rgb(p, q, h + 1 / 3) * 255))
    g = int(round(_hue2rgb(p, q, h) * 255))
    b = int(round(_hue2rgb(p, q, h - 1 / 3) * 255))
    return f"#{r:02x}{g:02x}{b:02x}"


def _weights_to_color(w_in: float, w_nb: float, w_wt: float) -> str:
    """
    Barycentric RYB color from three category weights.
    White = equal mix; corners = pure blue / yellow / red.
    """
    total = w_in + w_nb + w_wt
    if total < 1e-9:
        return "#cccccc"

    a = w_wt / total   # Battery Waste  → 0°  Red
    b = w_nb / total   # New Batteries  → 120° Yellow
    c = w_in / total   # Battery Inputs → 240° Blue

    vx = (a * math.cos(_RYB_RAD["Battery Waste"])
          + b * math.cos(_RYB_RAD["New Batteries"])
          + c * math.cos(_RYB_RAD["Battery Inputs"]))
    vy = (a * math.sin(_RYB_RAD["Battery Waste"])
          + b * math.sin(_RYB_RAD["New Batteries"])
          + c * math.sin(_RYB_RAD["Battery Inputs"]))

    magnitude = math.sqrt(vx ** 2 + vy ** 2)
    if magnitude < 1e-6:
        return "#ffffff"

    ryb_deg = math.degrees(math.atan2(vy, vx)) % 360
    rgb_hue = _ryb_to_rgb_hue(ryb_deg)
    saturation = magnitude ** 0.5   # gamma → richer intermediate colors
    lightness = 1.0 - 0.5 * saturation

    return _hsl_to_hex(rgb_hue, saturation, lightness)


# ── Color triangle legend ─────────────────────────────────────────────────────

def _make_color_triangle_fig() -> go.Figure:
    """Return a compact ternary scatter showing the RYB color space as a legend."""
    n = 25
    pts: list[dict] = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            a_in  = k / n   # Battery Inputs share
            a_nb  = j / n   # New Batteries share
            a_wt  = i / n   # Battery Waste share
            color = _weights_to_color(a_in, a_nb, a_wt)
            pts.append({
                "a": a_in * 100,
                "b": a_nb * 100,
                "c": a_wt * 100,
                "color": color,
                "label": (
                    f"Battery Inputs: {a_in:.0%}<br>"
                    f"New Batteries: {a_nb:.0%}<br>"
                    f"Battery Waste: {a_wt:.0%}"
                ),
            })
    fig = go.Figure(go.Scatterternary(
        a=[p["a"] for p in pts],
        b=[p["b"] for p in pts],
        c=[p["c"] for p in pts],
        mode="markers",
        marker=dict(
            color=[p["color"] for p in pts],
            size=12,
            symbol="square",
            line=dict(width=0),
        ),
        text=[p["label"] for p in pts],
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    ))
    fig.update_layout(
        ternary=dict(
            sum=100,
            aaxis=dict(
                title=dict(text="Battery Inputs", font=dict(size=11, color="#1a3a8f")),
                min=0, linewidth=1, gridcolor="#ccc", tickfont=dict(size=9),
            ),
            baxis=dict(
                title=dict(text="New Batteries", font=dict(size=11, color="#c9920a")),
                min=0, linewidth=1, gridcolor="#ccc", tickfont=dict(size=9),
            ),
            caxis=dict(
                title=dict(text="Battery Waste", font=dict(size=11, color="#b71c1c")),
                min=0, linewidth=1, gridcolor="#ccc", tickfont=dict(size=9),
            ),
            bgcolor="#fafafa",
        ),
        margin=dict(l=60, r=60, t=30, b=30),
        height=280,
        title=dict(text="Color key — trade composition", font=dict(size=12), x=0.5),
    )
    return fig


# ── Data aggregation ──────────────────────────────────────────────────────────

def _compute_totals(
    baci: pd.DataFrame,
    direction: str,
    year: int,
    pb_factors: dict[int, float],
) -> pd.DataFrame:
    """Return country | Battery Inputs | New Batteries | Battery Waste (tonnes Pb)."""
    rel = set(_HS_TO_CAT)
    df = baci[(baci["Year"] == year) & (baci["Product"].isin(rel))].copy()
    if df.empty:
        return pd.DataFrame(columns=["country"] + _CATEGORIES)

    df["pb_t"] = df["Product"].map(pb_factors).fillna(0.70) * df["Quantity"]
    df["category"] = df["Product"].map(_HS_TO_CAT)

    country_col = "Importer" if direction == "Imports" else "Exporter"
    out = (
        df.groupby([country_col, "category"])["pb_t"]
        .sum().unstack(fill_value=0.0)
        .reset_index()
        .rename(columns={country_col: "country"})
    )
    for cat in _CATEGORIES:
        if cat not in out.columns:
            out[cat] = 0.0
    return out[["country"] + _CATEGORIES]


# ── BACI country name → ISO-3 ─────────────────────────────────────────────────
_BACI_TO_ISO3: dict[str, str] = {
    "Afghanistan": "AFG", "Albania": "ALB", "Algeria": "DZA",
    "Angola": "AGO", "Argentina": "ARG", "Armenia": "ARM",
    "Australia": "AUS", "Austria": "AUT", "Azerbaijan": "AZE",
    "Bahrain": "BHR", "Bangladesh": "BGD", "Belarus": "BLR",
    "Belgium": "BEL", "Benin": "BEN",
    "Bolivia (Plurinational State of)": "BOL", "Bolivia": "BOL",
    "Bosnia Herzegovina": "BIH", "Brazil": "BRA", "Bulgaria": "BGR",
    "Burkina Faso": "BFA", "Cambodia": "KHM", "Cameroon": "CMR",
    "Canada": "CAN", "Chile": "CHL", "China": "CHN", "Colombia": "COL",
    "Congo": "COG", "Costa Rica": "CRI", "Côte d'Ivoire": "CIV",
    "Croatia": "HRV", "Cuba": "CUB", "Cyprus": "CYP", "Czechia": "CZE",
    "Dem. People's Rep. of Korea": "PRK", "Dem. Rep. of the Congo": "COD",
    "Denmark": "DNK", "Dominican Rep.": "DOM", "Ecuador": "ECU",
    "Egypt": "EGY", "El Salvador": "SLV", "Estonia": "EST",
    "Ethiopia": "ETH", "Finland": "FIN", "France": "FRA",
    "Gabon": "GAB", "Georgia": "GEO", "Germany": "DEU",
    "Ghana": "GHA", "Greece": "GRC", "Guatemala": "GTM",
    "Guinea": "GIN", "Honduras": "HND", "Hong Kong, China": "HKG",
    "Hungary": "HUN", "India": "IND", "Indonesia": "IDN",
    "Iran": "IRN", "Iraq": "IRQ", "Ireland": "IRL",
    "Israel": "ISR", "Italy": "ITA", "Jamaica": "JAM",
    "Japan": "JPN", "Jordan": "JOR", "Kazakhstan": "KAZ",
    "Kenya": "KEN", "Kuwait": "KWT", "Kyrgyzstan": "KGZ",
    "Lao People's Dem. Rep.": "LAO", "Latvia": "LVA", "Lebanon": "LBN",
    "Liberia": "LBR", "Lithuania": "LTU", "Luxembourg": "LUX",
    "Malaysia": "MYS", "Mali": "MLI", "Mexico": "MEX",
    "Morocco": "MAR", "Mozambique": "MOZ", "Myanmar": "MMR",
    "Netherlands": "NLD", "New Zealand": "NZL", "Nigeria": "NGA",
    "North Macedonia": "MKD", "Norway": "NOR", "Oman": "OMN",
    "Pakistan": "PAK", "Panama": "PAN", "Peru": "PER",
    "Philippines": "PHL", "Poland": "POL", "Portugal": "PRT",
    "Qatar": "QAT", "Rep. of Korea": "KOR", "Rep. of Moldova": "MDA",
    "Romania": "ROU", "Russian Federation": "RUS", "Saudi Arabia": "SAU",
    "Senegal": "SEN", "Serbia": "SRB", "Sierra Leone": "SLE",
    "Singapore": "SGP", "Slovakia": "SVK", "Slovenia": "SVN",
    "South Africa": "ZAF", "Spain": "ESP", "Sri Lanka": "LKA",
    "Sudan": "SDN", "Sweden": "SWE", "Switzerland": "CHE",
    "Taiwan": "TWN", "Tanzania": "TZA", "Thailand": "THA",
    "Togo": "TGO", "Trinidad and Tobago": "TTO", "Tunisia": "TUN",
    "Türkiye": "TUR", "Uganda": "UGA", "Ukraine": "UKR",
    "United Arab Emirates": "ARE", "United Kingdom": "GBR",
    "USA": "USA", "Uruguay": "URY", "Uzbekistan": "UZB",
    "Venezuela": "VEN", "Viet Nam": "VNM", "Yemen": "YEM",
    "Zambia": "ZMB", "Zimbabwe": "ZWE",
    "Eritrea": "ERI", "Djibouti": "DJI", "Gambia": "GMB",
    "Guinea-Bissau": "GNB", "Libya": "LBY", "Madagascar": "MDG",
    "Malawi": "MWI", "Mauritania": "MRT", "Mauritius": "MUS",
    "Niger": "NER", "Rwanda": "RWA", "Somalia": "SOM",
    "South Sudan": "SSD", "Eswatini": "SWZ", "Chad": "TCD",
    "Central African Rep.": "CAF", "Nepal": "NPL", "Mongolia": "MNG",
    "Papua New Guinea": "PNG", "Malta": "MLT", "Iceland": "ISL",
    "Brunei Darussalam": "BRN", "New Caledonia": "NCL",
    "Palestine": "PSE", "Syrian Arab Republic": "SYR",
    "North Korea": "PRK", "South Korea": "KOR",
    "Macau, China": "MAC", "Hong Kong": "HKG",
    "Mozambique": "MOZ", "Namibia": "NAM", "Botswana": "BWA",
    "Lesotho": "LSO", "Swaziland": "SWZ", "Comoros": "COM",
    "Cape Verde": "CPV", "Cabo Verde": "CPV", "Sao Tome and Principe": "STP",
    "Equatorial Guinea": "GNQ", "Seychelles": "SYC",
    "Afghanistan": "AFG", "Bhutan": "BTN", "Maldives": "MDV",
    "Myanmar": "MMR", "Timor-Leste": "TLS",
    "Fiji": "FJI", "Solomon Islands": "SLB", "Vanuatu": "VUT",
    "Samoa": "WSM", "Tonga": "TON", "Kiribati": "KIR",
    "Micronesia": "FSM", "Palau": "PLW", "Marshall Islands": "MHL",
    "Nauru": "NRU", "Tuvalu": "TUV",
    "Antigua and Barbuda": "ATG", "Bahamas": "BHS",
    "Barbados": "BRB", "Belize": "BLZ", "Dominica": "DMA",
    "Grenada": "GRD", "Guyana": "GUY", "Haiti": "HTI",
    "Nicaragua": "NIC", "Paraguay": "PRY",
    "Saint Kitts and Nevis": "KNA", "Saint Lucia": "LCA",
    "Saint Vincent and the Grenadines": "VCT", "Suriname": "SUR",
    "Trinidad and Tobago": "TTO",
    "Albania": "ALB", "Andorra": "AND", "Belarus": "BLR",
    "Bosnia Herzegovina": "BIH", "Estonia": "EST",
    "Kosovo": "XKX", "Latvia": "LVA", "Liechtenstein": "LIE",
    "Lithuania": "LTU", "Moldova": "MDA", "Monaco": "MCO",
    "Montenegro": "MNE", "North Macedonia": "MKD",
    "San Marino": "SMR", "Slovakia": "SVK",
    "Armenia": "ARM", "Georgia": "GEO",
    "Tajikistan": "TJK", "Turkmenistan": "TKM",
    "Other Asia, nes": None, "Areas, nes": None,
    "Free Zones": None, "Neutral Zone": None,
}


# 5 distinct colors for up to 5 selected countries
_PALETTE = ["#1565C0", "#2E7D32", "#E65100", "#6A1B9A", "#00838F"]


def _country_trade(
    baci: pd.DataFrame,
    country: str,
    year: int,
    pb_factors: dict[int, float],
) -> tuple[dict[str, float], dict[str, float]]:
    """Return (imports_tonnes, exports_tonnes) dicts for all three categories."""
    rel = set(_HS_TO_CAT)

    def _agg(country_col: str) -> dict[str, float]:
        df = baci[
            (baci["Year"] == year)
            & (baci["Product"].isin(rel))
            & (baci[country_col] == country)
        ].copy()
        if df.empty:
            return {cat: 0.0 for cat in _CATEGORIES}
        df["pb_t"] = df["Product"].map(pb_factors).fillna(0.70) * df["Quantity"]
        df["category"] = df["Product"].map(_HS_TO_CAT)
        grouped = df.groupby("category")["pb_t"].sum()
        return {cat: float(grouped.get(cat, 0.0)) for cat in _CATEGORIES}

    return _agg("Importer"), _agg("Exporter")


def _multi_ternary_chart(
    entries: list[tuple[str, dict[str, float], dict[str, float]]],
    year: int,
) -> go.Figure:
    """
    Ternary plot for multiple countries (up to 5).

    Each country gets:
      • An open circle at the imports position
      • A filled triangle at the exports position
      • A dashed line connecting them (acting as the arrow shaft)

    Axes: Battery Inputs (top) · New Batteries (left) · Battery Waste (right)
    """
    def _pct(d: dict[str, float]):
        total = sum(d.values())
        if total < 1e-9:
            return None
        return tuple(100 * d[cat] / total for cat in _CATEGORIES)

    fig = go.Figure()

    for idx, (country, imp, exp) in enumerate(entries):
        color = _PALETTE[idx % len(_PALETTE)]
        imp_pct = _pct(imp)
        exp_pct = _pct(exp)

        if imp_pct and exp_pct:
            fig.add_trace(go.Scatterternary(
                a=[imp_pct[0], exp_pct[0]],
                b=[imp_pct[1], exp_pct[1]],
                c=[imp_pct[2], exp_pct[2]],
                mode="lines",
                line=dict(color=color, width=2, dash="dash"),
                showlegend=False,
                hoverinfo="skip",
            ))

        if imp_pct:
            a, b, c = imp_pct
            fig.add_trace(go.Scatterternary(
                a=[a], b=[b], c=[c],
                mode="markers+text",
                name=f"{country} — Imports",
                text=[country],
                textposition="top center",
                textfont=dict(size=10, color=color),
                marker=dict(
                    size=13, color="white", symbol="circle",
                    line=dict(color=color, width=2.5),
                ),
                hovertemplate=(
                    f"<b>{country} — Imports</b><br>"
                    f"Battery Inputs: {a:.1f}%  ({imp['Battery Inputs']/1000:.1f} kt Pb)<br>"
                    f"New Batteries: {b:.1f}%  ({imp['New Batteries']/1000:.1f} kt Pb)<br>"
                    f"Battery Waste: {c:.1f}%  ({imp['Battery Waste']/1000:.1f} kt Pb)"
                    "<extra></extra>"
                ),
            ))

        if exp_pct:
            a, b, c = exp_pct
            fig.add_trace(go.Scatterternary(
                a=[a], b=[b], c=[c],
                mode="markers",
                name=f"{country} — Exports",
                marker=dict(
                    size=13, color=color, symbol="triangle-up",
                    line=dict(color="white", width=1.5),
                ),
                hovertemplate=(
                    f"<b>{country} — Exports</b><br>"
                    f"Battery Inputs: {a:.1f}%  ({exp['Battery Inputs']/1000:.1f} kt Pb)<br>"
                    f"New Batteries: {b:.1f}%  ({exp['New Batteries']/1000:.1f} kt Pb)<br>"
                    f"Battery Waste: {c:.1f}%  ({exp['Battery Waste']/1000:.1f} kt Pb)"
                    "<extra></extra>"
                ),
            ))

    fig.update_layout(
        title=dict(text=f"Import vs Export Composition — {year}", font_size=15, x=0.5, xanchor="center"),
        ternary=_ternary_axes(),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5, font=dict(size=11)),
        margin=dict(l=60, r=60, t=70, b=20),
        height=520,
    )
    return fig


def _net_ternary_chart(
    entries: list[tuple[str, dict[str, float], dict[str, float]]],
    year: int,
) -> go.Figure:
    """
    Ternary plot showing net trade position — one point per country.

    Position is determined by the relative share of net-import categories:
      coordinate_i = max(0, imports_i − exports_i)  normalized to sum to 100.

    Countries that are net exporters in all three categories cannot be placed
    and are omitted (noted in the hover and caption).
    """
    fig = go.Figure()
    skipped: list[str] = []

    for idx, (country, imp, exp) in enumerate(entries):
        color = _PALETTE[idx % len(_PALETTE)]

        nets = {cat: imp[cat] - exp[cat] for cat in _CATEGORIES}
        pull = {cat: max(0.0, nets[cat]) for cat in _CATEGORIES}
        total_pull = sum(pull.values())

        if total_pull < 1e-9:
            skipped.append(country)
            continue

        a = 100 * pull["Battery Inputs"] / total_pull
        b = 100 * pull["New Batteries"] / total_pull
        c = 100 * pull["Battery Waste"] / total_pull

        def _sign(v: float) -> str:
            return f"+{v/1000:.1f}" if v >= 0 else f"{v/1000:.1f}"

        fig.add_trace(go.Scatterternary(
            a=[a], b=[b], c=[c],
            mode="markers+text",
            name=country,
            text=[country],
            textposition="top center",
            textfont=dict(size=10, color=color),
            marker=dict(
                size=14, color=color, symbol="circle",
                line=dict(color="white", width=1.5),
            ),
            hovertemplate=(
                f"<b>{country} — Net Trade Position</b><br>"
                f"Battery Inputs: {_sign(nets['Battery Inputs'])} kt Pb<br>"
                f"New Batteries: {_sign(nets['New Batteries'])} kt Pb<br>"
                f"Battery Waste: {_sign(nets['Battery Waste'])} kt Pb<br>"
                "<i>Position set by net-import categories only</i>"
                "<extra></extra>"
            ),
        ))

    if skipped:
        fig.add_annotation(
            text=f"Net exporter (all categories) — not shown: {', '.join(skipped)}",
            xref="paper", yref="paper", x=0.5, y=-0.07,
            showarrow=False, font=dict(size=10, color="#888"),
            xanchor="center",
        )

    fig.update_layout(
        title=dict(text=f"Net Trade Position — {year}", font_size=15, x=0.5, xanchor="center"),
        ternary=_ternary_axes(),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5, font=dict(size=11)),
        margin=dict(l=60, r=60, t=70, b=30),
        height=520,
    )
    return fig


def _ternary_axes() -> dict:
    """Shared ternary axis layout used by both chart functions."""
    return dict(
        sum=100,
        aaxis=dict(title="Battery Inputs", min=0, linewidth=2, gridcolor="#ddd", tickfont=dict(size=10)),
        baxis=dict(title="New Batteries", min=0, linewidth=2, gridcolor="#ddd", tickfont=dict(size=10)),
        caxis=dict(title="Battery Waste", min=0, linewidth=2, gridcolor="#ddd", tickfont=dict(size=10)),
        bgcolor="#fafafa",
    )


# ── Main render ───────────────────────────────────────────────────────────────

def render_trade_composition_map(
    sidebar_year: int = 2022,
    dataset: str = "hs12",
    pb_factors: dict[int, float] | None = None,
) -> None:
    """Render the global trade composition map tab."""
    baci = load_master_baci(dataset)
    avail_years = sorted(int(y) for y in baci["Year"].unique())
    year = sidebar_year if sidebar_year in avail_years else max(avail_years)

    if year == 2024:
        st.warning("⚠ 2024 data is incomplete.")

    if pb_factors is None:
        pb_factors = {hs: meta["default"] for hs, meta in HS_META.items()}

    # ── Country comparison ternary (above the map) ────────────────────────────
    all_baci_countries = sorted(
        set(baci[baci["Year"] == year]["Importer"].unique())
        | set(baci[baci["Year"] == year]["Exporter"].unique())
    )

    sel_col, mode_col = st.columns([3, 2])
    with sel_col:
        selected_countries = st.multiselect(
            "Compare countries (up to 5)",
            all_baci_countries,
            default=["Ghana", "USA", "Rep. of Korea"] if all(
                c in all_baci_countries for c in ["Ghana", "USA", "Rep. of Korea"]
            ) else [],
            max_selections=5,
            key="comp_map_countries",
        )
    if selected_countries:
        entries = [
            (c, *_country_trade(baci, c, year, pb_factors))
            for c in selected_countries
        ]

        st.plotly_chart(_multi_ternary_chart(entries, year), use_container_width=True)
        st.caption(
            "Open circle = imports composition · Filled triangle = exports composition · "
            "Dashed line = shift from imports to exports. "
            "Corners = pure single-category economy; center = equal balance across all three."
        )
        st.divider()

    # ── Choropleth map ────────────────────────────────────────────────────────
    col_dir, _ = st.columns([2, 5])
    with col_dir:
        direction = st.radio(
            "Map: trade direction", ["Imports", "Exports"],
            horizontal=True, key="comp_map_direction",
        )

    totals = _compute_totals(baci, direction, year, pb_factors)

    if totals.empty:
        st.warning(f"No trade data found for {year}.")
        return

    # Build per-country color rows
    rows: list[dict] = []
    for _, row in totals.iterrows():
        iso3 = _BACI_TO_ISO3.get(str(row["country"]))
        if not iso3:
            continue
        w_in = float(row["Battery Inputs"])
        w_nb = float(row["New Batteries"])
        w_wt = float(row["Battery Waste"])
        total = w_in + w_nb + w_wt
        if total < 1e-6:
            continue
        rows.append({
            "iso3": iso3,
            "country": row["country"],
            "color": _weights_to_color(w_in, w_nb, w_wt),
            "in_kt": w_in / 1000,
            "nb_kt": w_nb / 1000,
            "wt_kt": w_wt / 1000,
            "total_kt": total / 1000,
            "pct_in": 100 * w_in / total,
            "pct_nb": 100 * w_nb / total,
            "pct_wt": 100 * w_wt / total,
        })

    if not rows:
        st.warning("No countries could be mapped — check name lookup table.")
        return

    df = pd.DataFrame(rows).reset_index(drop=True)
    N = len(df)

    # Plotly choropleth with a discrete step colorscale (one color per country)
    colorscale: list[list] = []
    for i, color in enumerate(df["color"]):
        colorscale.append([i / N, color])
        colorscale.append([(i + 1) / N, color])

    fig = go.Figure(go.Choropleth(
        locations=df["iso3"],
        z=[i + 0.5 for i in range(N)],
        zmin=0,
        zmax=N,
        locationmode="ISO-3",
        colorscale=colorscale,
        showscale=False,
        marker_line_color="white",
        marker_line_width=0.5,
        customdata=df[["country", "pct_in", "pct_nb", "pct_wt",
                        "total_kt", "in_kt", "nb_kt", "wt_kt"]].values,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Battery Inputs: %{customdata[1]:.1f}%"
            " (%{customdata[5]:,.1f} kt Pb)<br>"
            "New Batteries: %{customdata[2]:.1f}%"
            " (%{customdata[6]:,.1f} kt Pb)<br>"
            "Battery Waste: %{customdata[3]:.1f}%"
            " (%{customdata[7]:,.1f} kt Pb)<br>"
            "Total: %{customdata[4]:,.1f} kt Pb"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        title=dict(
            text=f"Lead Trade Composition — {direction}, {year}",
            font_size=15,
        ),
        geo=dict(
            showframe=False,
            showcoastlines=True,
            coastlinecolor="#aaaaaa",
            projection_type="natural earth",
            bgcolor="#f0f4f8",
            landcolor="#e8ebe0",
            showland=True,
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        height=520,
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── Color key ─────────────────────────────────────────────────────────────
    st.caption(
        f"Color shows each country's {direction.lower()} mix across three categories "
        "(excluding ore and other lead products). "
        "Hover for percentages and volumes. "
        "Year set by the sidebar slider."
    )

    _key_map_col, _key_tri_col = st.columns([3, 2])
    with _key_map_col:
        key_html = """
<div style="display:flex;flex-wrap:wrap;gap:14px;margin-top:4px;align-items:center;
            font-size:13px;line-height:1.4;">
  <div style="display:flex;align-items:center;gap:6px;">
    <div style="width:16px;height:16px;background:#1a3a8f;border-radius:3px;flex-shrink:0;"></div>
    <span><b>Battery Inputs</b> — refined lead, oxides, battery parts<br>
    <span style="color:#666;font-size:11px;">HS 780110/191/199, 282410/490, 850790</span></span>
  </div>
  <div style="display:flex;align-items:center;gap:6px;">
    <div style="width:16px;height:16px;background:#c9920a;border-radius:3px;flex-shrink:0;"></div>
    <span><b>New Batteries</b><br>
    <span style="color:#666;font-size:11px;">HS 850710/720</span></span>
  </div>
  <div style="display:flex;align-items:center;gap:6px;">
    <div style="width:16px;height:16px;background:#b71c1c;border-radius:3px;flex-shrink:0;"></div>
    <span><b>Battery Waste</b> — scrap &amp; used batteries<br>
    <span style="color:#666;font-size:11px;">HS 780200, 854810/854911</span></span>
  </div>
  <div style="color:#666;font-size:12px;margin-top:4px;">
    Mixed colors: Blue+Yellow=Green &nbsp;·&nbsp; Yellow+Red=Orange &nbsp;·&nbsp;
    Blue+Red=Purple &nbsp;·&nbsp; Equal shares=White
  </div>
</div>
"""
        st.markdown(key_html, unsafe_allow_html=True)
    with _key_tri_col:
        st.plotly_chart(_make_color_triangle_fig(), use_container_width=True)
