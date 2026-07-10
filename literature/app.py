"""
Literature Stats tab for the Pb Action Toolkit.

A single, deliberately simple view over the structured literature datapoints
(data/literature_datapoints.csv):

  * a search box + a few filters (topic, geography, year, evidence type)
  * a tight, sortable, downloadable results table
  * click a row to read the full datapoint in context (value, geography,
    evidence type, verbatim quote, and source citation)

No mass-balance model jargon is exposed to the user: the internal
`model_variable` codes (gamma, beta, eta_*, ...) are mapped to plain-English
*topics* and never shown.
"""
from pathlib import Path
import re

import pandas as pd
import streamlit as st

# Data-submission feature (Google Sheet) is deferred — re-enable later by
# uncommenting this import and the render_submission_form() call in render().
# from literature.submit import render_submission_form

_ROOT = Path(__file__).resolve().parent.parent
_DATA = _ROOT / "data" / "literature_datapoints.csv"

# --------------------------------------------------------------------------
# Topic taxonomy — plain-English grouping shown to users in place of the
# internal model_variable codes. Primary map by model_variable; the catch-all
# "other" rows are bucketed by keyword rules on the variable_label (below).
# --------------------------------------------------------------------------
_TOPIC_BY_VAR = {
    "gamma": "Collection & ULAB volumes",
    "ulab_volume": "Collection & ULAB volumes",
    "delta": "Recycling efficiency",
    "eta_smelt_F": "Recycling efficiency",
    "eta_smelt_I": "Recycling efficiency",
    "eta_refine_F": "Recycling efficiency",
    "eta_break_F": "Recycling efficiency",
    "eta_break_I": "Recycling efficiency",
    "eta_mfg_F": "Recycling efficiency",
    "eta_mfg_I": "Recycling efficiency",
    "secondary_smelter_capacity": "Recycling capacity & scale",
    "phi_smelt_f": "Recycling capacity & scale",
    "phi_break_f": "Recycling capacity & scale",
    "phi_refine_f": "Recycling capacity & scale",
    "phi_mfg_f": "Recycling capacity & scale",
    "battery_pb_content": "Battery composition & weight",
    "pb_mass_fraction": "Battery composition & weight",
    "battery_per_vehicle": "Battery composition & weight",
    "tau": "Battery lifespan",
    "sli_share": "Lead demand & consumption",
    "beta": "Lead demand & consumption",
}

# (keyword, topic) — first match wins; applied to the lower-cased label of
# rows whose model_variable is "other". Order matters (health before trade,
# capacity before consumption, etc.).
_TOPIC_KEYWORDS = [
    (("bll", "blood lead", "soil lead", "water lead", "air lead", "air dispersion",
      "emission", "deaths", "pollution", "exposed", "exposure", "education",
      "earnings", "score decline", "chelation", "ug/dl", "µg", "poisoning"),
     "Health & environmental exposure"),
    (("export", "import", "slab", "canada", "us-origin", "us share", "share of us",
      "shipped", "trade"),
     "Trade flows"),
    (("off-grid", "mini-grid", "minigrid", "solar", "gogla"),
     "Off-grid solar & mini-grids"),
    (("cost", "investment", "capital", "subsidy", "value per", "operating",
      "recoverable lead value"),
     "Recycling economics & cost"),
    (("efficiency", "recovery", "purity", "furnace", "fumes", "slag", "rotary",
      "blast", "pyrometall", "hydrometall", "electrowinning", "whr", "loss"),
     "Recycling efficiency"),
    (("capacity", "minimum scale", "minimum viable", "plant size", "plants",
      "standard", "sites", "count", "scale", "companies"),
     "Recycling capacity & scale"),
    (("consumption", "lead demand", "lead usage", "secondary lead share",
      "secondary (recycled) lead", "recycled lead", "refined lead usage",
      "share of supply", "global lead"),
     "Lead demand & consumption"),
    (("battery weight", "car battery", "motorcycle battery", "lead content",
      "lead per", "battery lead", "battery drainage"),
     "Battery composition & weight"),
    (("lifespan", "lifetime"),
     "Battery lifespan"),
]
_TOPIC_ORDER = [
    "Collection & ULAB volumes", "Recycling efficiency", "Recycling capacity & scale",
    "Recycling economics & cost", "Battery composition & weight", "Battery lifespan",
    "Lead demand & consumption", "Trade flows", "Health & environmental exposure",
    "Off-grid solar & mini-grids", "Other",
]


def _topic(model_variable: str, label: str) -> str:
    if model_variable in _TOPIC_BY_VAR:
        return _TOPIC_BY_VAR[model_variable]
    lbl = (label or "").lower()
    for keywords, topic in _TOPIC_KEYWORDS:
        if any(k in lbl for k in keywords):
            return topic
    return "Other"


def _type_group(t: str) -> str:
    """Collapse the ~17 raw evidence-type strings into a handful of buckets."""
    if not isinstance(t, str) or not t.strip():
        return "Unspecified"
    base = re.split(r"[/(]", t)[0].strip()
    if base.lower().startswith("model"):
        return "Modeled"
    return base or "Unspecified"


def _year_num(y):
    """First 4-digit year found in the (often messy) year string, else None."""
    if pd.isna(y):
        return None
    m = re.search(r"(19|20)\d{2}", str(y))
    return int(m.group(0)) if m else None


def _value_str(r) -> str:
    """Compact 'as-reported' value with units."""
    if isinstance(r.value_text, str) and r.value_text.strip():
        v = r.value_text.strip()
    elif pd.notna(r.value):
        if pd.notna(r.value_low) and pd.notna(r.value_high):
            v = f"{r.value_low:g}-{r.value_high:g}"
        else:
            v = f"{r.value:g}"
    else:
        v = "-"
    if isinstance(r.units, str) and r.units.strip():
        v = f"{v} {r.units.strip()}"
    return v


@st.cache_data(show_spinner=False)
def _load() -> pd.DataFrame:
    df = pd.read_csv(_DATA)
    df["topic"] = [_topic(mv, lbl) for mv, lbl in zip(df["model_variable"], df["variable_label"])]
    df["evidence"] = df["type"].apply(_type_group)
    df["geography"] = (df["country_region"].fillna("")
                       .str.replace(r"\s*\(.*?\)", "", regex=True).str.strip()
                       .replace("", "Unspecified"))
    df["year_num"] = df["year"].apply(_year_num)
    df["value_display"] = [_value_str(r) for r in df.itertuples()]
    return df


# --------------------------------------------------------------------------
def render() -> None:
    """Public entry point: the searchable literature library.

    The data-submission form is deferred for now — re-enable it later by
    uncommenting the render_submission_form() call below (and its import above)."""
    _render_library()
    # render_submission_form()


def _render_library() -> None:
    st.title("Literature Stats")
    st.write(
        "A searchable library of quantitative datapoints pulled from the "
        "lead-battery literature. Use the search box and filters to "
        "find figures, then click any row to read it in context and see its source."
    )

    st.warning(
        "Always open the cited source and read the surrounding text "
        "before quoting or building on a number. Treat this as a starting point "
        "for research, not a settled answer.",
        icon="⚠️",
    )

    try:
        df = _load()
    except FileNotFoundError:
        st.error("literature_datapoints.csv not found in data/.")
        return

    # ---- filters -------------------------------------------------------
    f1, f2 = st.columns([3, 2])
    q = f1.text_input(
        "Search", key="lit_q",
        placeholder="Search statistic, country, quote, source, notes...",
    ).strip().lower()
    topics = [t for t in _TOPIC_ORDER if t in set(df["topic"])]
    tsel = f2.multiselect("Topic", topics, key="lit_topic")

    f3, f4 = st.columns(2)
    geos = sorted(g for g in df["geography"].unique() if g and g != "Unspecified")
    gsel = f3.multiselect("Geography", geos, key="lit_geo")
    evs = sorted(df["evidence"].unique())
    esel = f4.multiselect("Evidence type", evs, key="lit_evidence")

    yrs = df["year_num"].dropna()
    if not yrs.empty:
        ymin, ymax = int(yrs.min()), int(yrs.max())
        y1, y2 = st.columns([3, 2])
        if ymin < ymax:
            yr_range = y1.slider("Year", ymin, ymax, (ymin, ymax), key="lit_year")
        else:
            yr_range = (ymin, ymax)
        inc_undated = y2.checkbox("Include undated / multi-year", value=True, key="lit_undated")
    else:
        yr_range, inc_undated = None, True

    # ---- apply ---------------------------------------------------------
    view = df
    if tsel:
        view = view[view["topic"].isin(tsel)]
    if gsel:
        view = view[view["geography"].isin(gsel)]
    if esel:
        view = view[view["evidence"].isin(esel)]
    if yr_range is not None:
        in_range = view["year_num"].between(yr_range[0], yr_range[1])
        undated = view["year_num"].isna()
        view = view[in_range | (undated & inc_undated)]
    if q:
        hay = (view["variable_label"].fillna("") + " " + view["country_region"].fillna("") + " "
               + view["value_text"].fillna("") + " " + view["quote"].fillna("") + " "
               + view["citation_apa"].fillna("") + " " + view["notes"].fillna("")).str.lower()
        view = view[hay.str.contains(re.escape(q), regex=True)]

    view = view.reset_index(drop=True)

    st.markdown(f"### {len(view)} statistic(s)")
    if len(view) == 0:
        st.info("No statistics match. Clear the search or broaden the filters.")
        return

    # ---- results table (tight, sortable, selectable) -------------------
    table = pd.DataFrame({
        "Statistic": view["variable_label"],
        "Value": view["value_display"],
        "Geography": view["country_region"].fillna("-"),
        "Year": view["year"].fillna("-").astype(str),
        "Topic": view["topic"],
        "Evidence": view["evidence"],
    })

    event = st.dataframe(
        table, hide_index=True, width="stretch", height=430,
        on_select="rerun", selection_mode="single-row", key="lit_table",
        column_config={
            "Statistic": st.column_config.TextColumn(width="large"),
            "Value": st.column_config.TextColumn(width="medium"),
        },
    )

    # ---- download (friendly columns, no internal codes / pdf name) -----
    dl = pd.DataFrame({
        "statistic": view["variable_label"],
        "value_text": view["value_text"],
        "value": view["value"], "value_low": view["value_low"], "value_high": view["value_high"],
        "units": view["units"], "geography": view["country_region"], "year": view["year"],
        "topic": view["topic"], "evidence_type": view["type"],
        "source": view["citation_apa"], "page": view["page"], "quote": view["quote"],
        "notes": view["notes"],
    })
    st.download_button(
        "Download these statistics (CSV)",
        dl.to_csv(index=False).encode("utf-8"),
        "pb_action_literature_statistics.csv", "text/csv",
    )

    # ---- detail panel for the selected row -----------------------------
    rows = event.selection.rows if event and event.selection else []
    if not rows:
        st.caption("Click a row above to read the full statistic in context.")
        return

    r = view.iloc[rows[0]]
    st.divider()
    with st.container(border=True):
        st.markdown(f"#### {r['variable_label']}")
        d = st.columns(3)
        d[0].markdown(f"**Value**  \n{r['value_display']}")
        d[1].markdown(f"**Geography**  \n{r['country_region'] if isinstance(r['country_region'], str) else '-'}")
        d[2].markdown(f"**Evidence type**  \n{r['type'] if isinstance(r['type'], str) else '-'}"
                      + (f"  ·  {r['year']}" if pd.notna(r['year']) else ""))
        if isinstance(r["quote"], str) and r["quote"].strip():
            st.markdown("**In the source's words**")
            st.markdown(f"> *“{r['quote'].strip()}”*")
        src = r["citation_apa"] if isinstance(r["citation_apa"], str) else "Source not recorded"
        pg = f"  ·  {r['page']}" if isinstance(r["page"], str) and r["page"].strip() else ""
        st.markdown(f"**Source**  \n{src}{pg}")
        if isinstance(r["notes"], str) and r["notes"].strip():
            st.caption(f"Notes: {r['notes']}")
