"""
data_loader.py — Load and preprocess all data files for the Pb Action Lead Usage Model (V4).

All mass quantities in metric tonnes of lead content.

Files loaded:
  BACI_lead_trade_2012_2024_modified_vHS_4.csv  — bilateral trade data
  installation_estimates.csv                    — exogenous installation estimates
  mining_production.csv                         — mine production by country
  country_parameters.csv                        — per-country model parameters (V4)
  lead_smelter_capacity_by_country_2012_2024.csv — smelter capacity (D1 diagnostic only)
  continent_defaults.csv                        — fallback parameters by continent
"""

from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"

# HS code → model category
CATEGORY_MAP: dict[int, str] = {
    260700: "ORE",
    780110: "FEED",
    780191: "FEED",
    780199: "FEED",
    850790: "FEED",
    282410: "FEED",
    282490: "FEED",
    850710: "BATT",
    850720: "BATT",
    854810: "USED",   # HS12
    854911: "USED",   # HS22 equivalent of 854810
    780200: "SCRAP",
}

# Pb conversion factors applied to raw Quantity → actual_lead (tonnes Pb).
# Used when loading HS22 data, which has no pre-computed Pb Quantity column.
# The keys of this dict also define which HS codes are KEPT when loading HS22
# (see load_baci); every code the app can display must appear here. The Slag
# (262021/262029) and Other Lead Products (780411/780419/780420/780600) codes
# are populated in the HS22 file and are retained here so they survive the load
# (they are still excluded by default in the UI). Their factors match the
# HS_META defaults used by the material-flow tabs.
HS_LEAD_FACTORS: dict[int, float] = {
    260700: 0.60,
    780110: 1.00,
    780191: 0.95,
    780199: 0.95,
    780200: 0.97,
    850710: 0.65,
    850720: 0.70,
    850790: 0.80,
    854810: 0.70,
    854911: 0.70,
    282410: 0.91,
    282490: 0.75,
    # Slag (default-off in the UI)
    262021: 0.40,
    262029: 0.55,
    # Other Lead Products (default-off in the UI)
    780411: 0.99,
    780419: 0.99,
    780420: 0.985,
    780600: 0.97,
}

# Countries for which Eurostat lead-acid battery collection data exists
# (derived from Collect_data.csv — uses BACI country names directly)
EUROSTAT_ELIGIBLE: frozenset[str] = frozenset({
    "Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czechia",
    "Denmark", "Estonia", "Finland", "France", "Germany", "Greece",
    "Hungary", "Iceland", "Ireland", "Italy", "Latvia", "Lithuania",
    "Luxembourg", "Malta", "Netherlands", "Norway", "Poland", "Portugal",
    "Romania", "Slovakia", "Slovenia", "Spain", "Sweden",
    # Non-EU countries present in Collect_data.csv:
    "Ghana", "Thailand",
})

# Refining hubs that process significant non-EU imported feedstock;
# Eurostat anchor understates their total smelting input.
REFINING_HUBS: frozenset[str] = frozenset({"Netherlands", "Belgium", "Germany"})

# Türkiye name fix (garbled due to encoding in raw BACI)
_COUNTRY_NAME_FIXES: dict[str, str] = {
    "TÃƒÂ¼rkiye": "Türkiye",
    "TÃ¼rkiye": "Türkiye",
}


def load_baci(dataset: str = "hs12") -> pd.DataFrame:
    """
    Load BACI trade data.

    Parameters
    ----------
    dataset : "hs12" (default) or "hs22"
        "hs12" loads the full 2012–2024 HS17-coded file with a pre-computed
        Pb Quantity column.  "hs22" loads the 2022–2024 HS22-coded file and
        computes actual_lead on the fly using HS_LEAD_FACTORS.

    Returns DataFrame with columns:
        Year, Exporter, Importer, Product, Value, Quantity, actual_lead, category
    """
    if dataset == "hs22":
        df = pd.read_csv(DATA_DIR / "BACI_HS22_lead_trade_2022_2024.csv")
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
        # Keep every code the app knows how to convert (the HS_LEAD_FACTORS keys).
        # This includes Slag and Other Lead Products, which are present in the
        # HS22 file; codes outside this set (e.g. non-lead 293110/381111) are
        # dropped.
        model_codes = set(HS_LEAD_FACTORS.keys())
        df = df[df["Product"].isin(model_codes)].copy()
        df["actual_lead"] = df["Product"].map(HS_LEAD_FACTORS).fillna(0.70) * df["Quantity"]
    else:
        df = pd.read_csv(
            DATA_DIR / "BACI_lead_trade_2012_2024_modified_vHS_4.csv",
            encoding="utf-8-sig",
        )
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
        df = df.rename(columns={"Pb Quantity": "actual_lead"})

    df["category"] = df["Product"].map(CATEGORY_MAP)

    # Fix garbled country names
    for col in ("Exporter", "Importer"):
        df[col] = df[col].replace(_COUNTRY_NAME_FIXES)

    return df


def load_installation_estimates() -> pd.DataFrame:
    """
    Load exogenous battery installation estimates.

    Returns DataFrame with columns:
        Country, install_kt, install_t, Year, Method, Sources

    install_t is the primary quantity (metric tonnes Pb/yr).
    """
    df = pd.read_csv(DATA_DIR / "installation_estimates.csv")
    df = df.rename(columns={"Lead in Batteries Installed (kt Pb/yr)": "install_kt"})
    df["install_t"] = df["install_kt"] * 1000.0
    return df


def load_mining_production() -> pd.DataFrame:
    """
    Load mine production data.

    Returns DataFrame with columns:
        Country, mining_t, Year, Reserves (t), Sources

    mining_t is already in tonnes of lead content.
    """
    df = pd.read_csv(DATA_DIR / "mining_production.csv")
    df = df.rename(
        columns={
            "Mine Production Lead Content (t)": "mining_t",
            "Estimate Year": "Year",
        }
    )
    return df


def load_country_parameters() -> pd.DataFrame:
    """
    Load per-country model parameters (V4).

    Returns DataFrame with columns (at minimum):
        Country, BACI_name, income_group, continent,
        gamma, tau, beta,
        eta_break_formal, eta_break_informal,
        eta_scrap_formal, eta_scrap_informal, eta_mfg,
        mining_kt, SLI_pct,
        phi_break_default, phi_smelt_default, lambda_divert,
        informal_mode, peer_country,
        calibration_confidence, calibration_notes, notes
    """
    return pd.read_csv(DATA_DIR / "country_parameters.csv")


def load_smelter_capacity() -> pd.DataFrame:
    """
    Load secondary smelter capacity data.

    Returns DataFrame with columns:
        country, secondary_capacity_t_{year} for years 2012–2024

    Used ONLY for the D1 diagnostic. Not used in any model equation.

    Checks DATA_DIR/data/ first, then project root, to handle installations
    where this file sits alongside other CSVs rather than inside data/.
    """
    filename = "lead_smelter_capacity_by_country_2012_2024.csv"
    candidates = [
        DATA_DIR / filename,
        DATA_DIR.parent / filename,
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            if "Country" in df.columns and "country" not in df.columns:
                df = df.rename(columns={"Country": "country"})
            return df
    raise FileNotFoundError(
        f"Smelter capacity file not found. Searched: {[str(p) for p in candidates]}"
    )


# ── BACI name → Plotly "country names" locationmode mapping ──────────────────
# Only entries that DIFFER from Plotly's expected names are listed.
# Countries absent from this dict pass through unchanged.
BACI_TO_STANDARD_NAME: dict[str, str | None] = {
    # Major economies
    "USA":                              "United States",
    "Rep. of Korea":                    "South Korea",
    "Viet Nam":                         "Vietnam",
    "Russian Federation":               "Russia",
    "Türkiye":                          "Turkey",
    "Czechia":                          "Czech Republic",
    "Brunei Darussalam":                "Brunei",
    "Timor-Leste":                      "East Timor",
    "Cabo Verde":                       "Cape Verde",
    "Lao People's Dem. Rep.":           "Laos",
    "FS Micronesia":                    "Micronesia",
    "Dem. People's Rep. of Korea":      "North Korea",
    "Bolivia (Plurinational State of)": "Bolivia",
    "Venezuela":                        "Venezuela",
    # Congos
    "Dem. Rep. of the Congo":           "Democratic Republic of the Congo",
    "Congo":                            "Republic of the Congo",
    # Africa
    "Côte d'Ivoire":                    "Ivory Coast",
    "United Rep. of Tanzania":          "Tanzania",
    "Central African Rep.":             "Central African Republic",
    "Sao Tome and Principe":            "Sao Tome and Principe",
    "Eswatini":                         "Eswatini",
    # Europe / Caucasus
    "Bosnia Herzegovina":               "Bosnia and Herzegovina",
    "Rep. of Moldova":                  "Moldova",
    "North Macedonia":                  "North Macedonia",
    # Middle East
    "State of Palestine":               "Palestine",
    "Iran":                             "Iran",
    "Syria":                            "Syria",
    # Americas
    "Dominican Rep.":                   "Dominican Republic",
    "Falkland Isds (Malvinas)":         "Falkland Islands",
    "Cayman Isds":                      "Cayman Islands",
    "Turks and Caicos Isds":            "Turks and Caicos Islands",
    "Br. Virgin Isds":                  "British Virgin Islands",
    "Saint Barthélemy":                 "Saint Barthelemy",
    "Bonaire":                          "Bonaire, Saint Eustatius and Saba",
    # Pacific / Oceania
    "Solomon Isds":                     "Solomon Islands",
    "Marshall Isds":                    "Marshall Islands",
    "Cook Isds":                        "Cook Islands",
    "Wallis and Futuna Isds":           "Wallis and Futuna",
    "French Polynesia":                 "French Polynesia",
    "New Caledonia":                    "New Caledonia",
    "Mayotte (Overseas France)":        "Mayotte",
    # Not mappable to Plotly polygons → None (filtered out of map)
    "Other Asia":                       None,
    "Br. Indian Ocean Terr.":           None,
    "Christmas Isds":                   None,
    "Cocos Isds":                       None,
    "Pitcairn":                         None,
    "Tokelau":                          None,
    "Norfolk Isds":                     None,
    "Saint Pierre and Miquelon":        None,
}


def load_eurostat_collection(filepath=None) -> dict[tuple[str, int], float]:
    """
    Load Eurostat lead-acid battery collection data from Collect_data.csv.

    Returns a dict keyed by (country, year) → input_tonnes where input_tonnes
    is in tonnes of lead-acid battery weight (t LAB/yr), NOT tonnes of lead
    content. Apply a lead-content fraction (default 0.65) when converting.

    Returns an empty dict with a logged warning if the file is not found.
    """
    path = Path(filepath) if filepath else DATA_DIR / "Collect_data.csv"
    if not path.exists():
        import logging
        logging.warning(f"Eurostat collection file not found: {path}")
        return {}
    df = pd.read_csv(path)
    df = df[df["Variable"] == "COLLECT"].copy()
    result: dict[tuple[str, int], float] = {}
    for _, row in df.iterrows():
        try:
            result[(str(row["Country"]), int(row["Year"]))] = float(row["Value"])
        except (ValueError, KeyError):
            pass
    return result


def load_continent_defaults() -> pd.DataFrame:
    """Load continent-level default parameters from continent_defaults.csv."""
    return pd.read_csv(DATA_DIR / "continent_defaults.csv")


def load_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convenience loader. Returns (baci, installation, mining, params)."""
    return (
        load_baci(),
        load_installation_estimates(),
        load_mining_production(),
        load_country_parameters(),
    )


def load_all_with_capacity() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Extended convenience loader. Returns (baci, installation, mining, params, capacity).

    The capacity DataFrame is used only for the D1 diagnostic (Informal Smelting Gap)
    and is never passed to the model equations. Pass it to compute_all_diagnostics()
    via the capacity_df argument.

    Falls back gracefully: if the smelter capacity file is missing, returns None
    as the fifth element without raising an error.
    """
    baci, install, mining, params = load_all()
    try:
        capacity = load_smelter_capacity()
    except FileNotFoundError:
        capacity = None
    return baci, install, mining, params, capacity


# ── Country → Continent mapping (BACI country names) ─────────────────────────

_AFRICA = {
    "Algeria", "Angola", "Benin", "Botswana", "Br. Indian Ocean Terr.",
    "Burkina Faso", "Burundi", "Cabo Verde", "Cameroon", "Central African Rep.",
    "Chad", "Comoros", "Congo", "Côte d'Ivoire", "Dem. Rep. of the Congo",
    "Djibouti", "Egypt", "Equatorial Guinea", "Eritrea", "Eswatini", "Ethiopia",
    "Gabon", "Gambia", "Ghana", "Guinea", "Guinea-Bissau", "Kenya", "Lesotho",
    "Liberia", "Libya", "Madagascar", "Malawi", "Mali", "Mauritania", "Mauritius",
    "Mayotte (Overseas France)", "Morocco", "Mozambique", "Namibia", "Niger",
    "Nigeria", "Rwanda", "Saint Helena", "Sao Tome and Principe", "Senegal",
    "Seychelles", "Sierra Leone", "Somalia", "South Africa", "South Sudan",
    "Sudan", "Togo", "Tunisia", "Uganda", "United Rep. of Tanzania",
    "Zambia", "Zimbabwe",
}

_ASIA = {
    "Afghanistan", "Azerbaijan", "Bahrain", "Bangladesh", "Bhutan",
    "Brunei Darussalam", "Cambodia", "China", "Dem. People's Rep. of Korea",
    "India", "Indonesia", "Iran", "Iraq", "Israel", "Japan", "Jordan",
    "Kazakhstan", "Kuwait", "Kyrgyzstan", "Lao People's Dem. Rep.", "Lebanon",
    "Malaysia", "Maldives", "Mongolia", "Myanmar", "Nepal", "Oman", "Other Asia",
    "Pakistan", "Philippines", "Qatar", "Rep. of Korea", "Saudi Arabia",
    "Singapore", "Sri Lanka", "State of Palestine", "Syria", "Tajikistan",
    "Thailand", "Timor-Leste", "Türkiye", "Turkmenistan", "United Arab Emirates",
    "Uzbekistan", "Viet Nam", "Yemen",
}

_EUROPE = {
    "Albania", "Andorra", "Armenia", "Austria", "Belarus", "Belgium",
    "Bosnia Herzegovina", "Bulgaria", "Croatia", "Cyprus", "Czechia", "Denmark",
    "Estonia", "Finland", "France", "Georgia", "Germany", "Gibraltar", "Greece",
    "Hungary", "Iceland", "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg",
    "Malta", "Montenegro", "Netherlands", "North Macedonia", "Norway", "Poland",
    "Portugal", "Rep. of Moldova", "Romania", "Russian Federation", "San Marino",
    "Serbia", "Slovakia", "Slovenia", "Spain", "Sweden", "Switzerland",
    "Ukraine", "United Kingdom",
}

_AMERICAS = {
    "American Samoa", "Anguilla", "Antigua and Barbuda", "Argentina", "Aruba",
    "Bahamas", "Barbados", "Belize", "Bermuda", "Bolivia (Plurinational State of)",
    "Bonaire", "Brazil", "Canada", "Cayman Isds", "Chile", "Colombia",
    "Costa Rica", "Cuba", "Curaçao", "Dominica", "Dominican Rep.", "Ecuador",
    "El Salvador", "Falkland Isds (Malvinas)", "Greenland", "Grenada", "Guatemala",
    "Guyana", "Haiti", "Honduras", "Jamaica", "Mexico", "Montserrat",
    "Nicaragua", "Panama", "Paraguay", "Peru",
    "Saint Barthélemy", "Saint Kitts and Nevis", "Saint Lucia", "Saint Maarten",
    "Saint Pierre and Miquelon", "Saint Vincent and the Grenadines", "Suriname",
    "Trinidad and Tobago", "Turks and Caicos Isds", "Uruguay", "USA", "Venezuela",
}

# Oceania mapped to Americas (comparable development profiles for AU/NZ;
# Pacific island territories treated similarly)
_OCEANIA_AS_AMERICAS = {
    "Australia", "Christmas Isds", "Cocos Isds", "Cook Isds", "Fiji",
    "FS Micronesia", "French Polynesia", "Guam", "Kiribati", "Marshall Isds",
    "Nauru", "New Caledonia", "New Zealand", "Niue", "Norfolk Isds", "Palau",
    "Papua New Guinea", "Pitcairn", "Samoa", "Solomon Isds", "Tokelau", "Tonga",
    "Tuvalu", "Vanuatu", "Wallis and Futuna Isds",
}

COUNTRY_CONTINENT_MAP: dict[str, str] = (
    {c: "Africa" for c in _AFRICA}
    | {c: "Asia" for c in _ASIA}
    | {c: "Europe" for c in _EUROPE}
    | {c: "Americas" for c in _AMERICAS}
    | {c: "Americas" for c in _OCEANIA_AS_AMERICAS}
)


# ── Country → World Bank income group (BACI country names) ───────────────────
# Classifications: World Bank FY2024
# High income > $13,845 GNI/capita; Upper middle $4,466–$13,845;
# Lower middle $1,136–$4,465; Low income < $1,136
# Territories without WB classification assigned to nearest plausible group.

COUNTRY_INCOME_GROUP: dict[str, str] = {
    # ── Africa ────────────────────────────────────────────────────────────────
    "Algeria":                          "Lower middle income",
    "Angola":                           "Lower middle income",
    "Benin":                            "Low income",
    "Botswana":                         "Upper middle income",
    "Br. Indian Ocean Terr.":           "High income",
    "Burkina Faso":                     "Low income",
    "Burundi":                          "Low income",
    "Cabo Verde":                       "Lower middle income",
    "Cameroon":                         "Lower middle income",
    "Central African Rep.":             "Low income",
    "Chad":                             "Low income",
    "Comoros":                          "Lower middle income",
    "Congo":                            "Lower middle income",
    "Côte d'Ivoire":                    "Lower middle income",
    "Dem. Rep. of the Congo":           "Low income",
    "Djibouti":                         "Lower middle income",
    "Egypt":                            "Lower middle income",
    "Equatorial Guinea":                "Upper middle income",
    "Eritrea":                          "Low income",
    "Eswatini":                         "Lower middle income",
    "Ethiopia":                         "Low income",
    "Gabon":                            "Upper middle income",
    "Gambia":                           "Low income",
    "Ghana":                            "Lower middle income",
    "Guinea":                           "Low income",
    "Guinea-Bissau":                    "Low income",
    "Kenya":                            "Lower middle income",
    "Lesotho":                          "Lower middle income",
    "Liberia":                          "Low income",
    "Libya":                            "Upper middle income",
    "Madagascar":                       "Low income",
    "Malawi":                           "Low income",
    "Mali":                             "Low income",
    "Mauritania":                       "Lower middle income",
    "Mauritius":                        "Upper middle income",
    "Mayotte (Overseas France)":        "High income",
    "Morocco":                          "Lower middle income",
    "Mozambique":                       "Low income",
    "Namibia":                          "Upper middle income",
    "Niger":                            "Low income",
    "Nigeria":                          "Lower middle income",
    "Rwanda":                           "Low income",
    "Saint Helena":                     "High income",
    "Sao Tome and Principe":            "Lower middle income",
    "Senegal":                          "Lower middle income",
    "Seychelles":                       "High income",
    "Sierra Leone":                     "Low income",
    "Somalia":                          "Low income",
    "South Africa":                     "Upper middle income",
    "South Sudan":                      "Low income",
    "Sudan":                            "Low income",
    "Togo":                             "Low income",
    "Tunisia":                          "Lower middle income",
    "Uganda":                           "Low income",
    "United Rep. of Tanzania":          "Low income",
    "Zambia":                           "Lower middle income",
    "Zimbabwe":                         "Lower middle income",
    # ── Asia ─────────────────────────────────────────────────────────────────
    "Afghanistan":                      "Low income",
    "Azerbaijan":                       "Upper middle income",
    "Bahrain":                          "High income",
    "Bangladesh":                       "Lower middle income",
    "Bhutan":                           "Lower middle income",
    "Brunei Darussalam":                "High income",
    "Cambodia":                         "Lower middle income",
    "China":                            "Upper middle income",
    "Dem. People's Rep. of Korea":      "Low income",
    "India":                            "Lower middle income",
    "Indonesia":                        "Upper middle income",
    "Iran":                             "Lower middle income",
    "Iraq":                             "Upper middle income",
    "Israel":                           "High income",
    "Japan":                            "High income",
    "Jordan":                           "Upper middle income",
    "Kazakhstan":                       "Upper middle income",
    "Kuwait":                           "High income",
    "Kyrgyzstan":                       "Lower middle income",
    "Lao People's Dem. Rep.":           "Lower middle income",
    "Lebanon":                          "Lower middle income",
    "Malaysia":                         "Upper middle income",
    "Maldives":                         "Upper middle income",
    "Mongolia":                         "Lower middle income",
    "Myanmar":                          "Lower middle income",
    "Nepal":                            "Lower middle income",
    "Oman":                             "High income",
    "Other Asia":                       "Lower middle income",
    "Pakistan":                         "Lower middle income",
    "Philippines":                      "Lower middle income",
    "Qatar":                            "High income",
    "Rep. of Korea":                    "High income",
    "Saudi Arabia":                     "High income",
    "Singapore":                        "High income",
    "Sri Lanka":                        "Lower middle income",
    "State of Palestine":               "Lower middle income",
    "Syria":                            "Low income",
    "Tajikistan":                       "Lower middle income",
    "Thailand":                         "Upper middle income",
    "Timor-Leste":                      "Lower middle income",
    "Türkiye":                          "Upper middle income",
    "Turkmenistan":                     "Upper middle income",
    "United Arab Emirates":             "High income",
    "Uzbekistan":                       "Lower middle income",
    "Viet Nam":                         "Lower middle income",
    "Yemen":                            "Low income",
    # ── Europe ───────────────────────────────────────────────────────────────
    "Albania":                          "Upper middle income",
    "Andorra":                          "High income",
    "Armenia":                          "Upper middle income",
    "Austria":                          "High income",
    "Belarus":                          "Upper middle income",
    "Belgium":                          "High income",
    "Bosnia Herzegovina":               "Upper middle income",
    "Bulgaria":                         "Upper middle income",
    "Croatia":                          "High income",
    "Cyprus":                           "High income",
    "Czechia":                          "High income",
    "Denmark":                          "High income",
    "Estonia":                          "High income",
    "Finland":                          "High income",
    "France":                           "High income",
    "Georgia":                          "Upper middle income",
    "Germany":                          "High income",
    "Gibraltar":                        "High income",
    "Greece":                           "High income",
    "Hungary":                          "High income",
    "Iceland":                          "High income",
    "Ireland":                          "High income",
    "Italy":                            "High income",
    "Latvia":                           "High income",
    "Lithuania":                        "High income",
    "Luxembourg":                       "High income",
    "Malta":                            "High income",
    "Montenegro":                       "Upper middle income",
    "Netherlands":                      "High income",
    "North Macedonia":                  "Upper middle income",
    "Norway":                           "High income",
    "Poland":                           "High income",
    "Portugal":                         "High income",
    "Rep. of Moldova":                  "Lower middle income",
    "Romania":                          "High income",
    "Russian Federation":               "Upper middle income",
    "San Marino":                       "High income",
    "Serbia":                           "Upper middle income",
    "Slovakia":                         "High income",
    "Slovenia":                         "High income",
    "Spain":                            "High income",
    "Sweden":                           "High income",
    "Switzerland":                      "High income",
    "Ukraine":                          "Lower middle income",
    "United Kingdom":                   "High income",
    # ── Americas ─────────────────────────────────────────────────────────────
    "American Samoa":                   "Upper middle income",
    "Anguilla":                         "High income",
    "Antigua and Barbuda":              "High income",
    "Argentina":                        "Upper middle income",
    "Aruba":                            "High income",
    "Bahamas":                          "High income",
    "Barbados":                         "High income",
    "Belize":                           "Upper middle income",
    "Bermuda":                          "High income",
    "Bolivia (Plurinational State of)": "Lower middle income",
    "Bonaire":                          "High income",
    "Brazil":                           "Upper middle income",
    "Canada":                           "High income",
    "Cayman Isds":                      "High income",
    "Chile":                            "High income",
    "Colombia":                         "Upper middle income",
    "Costa Rica":                       "Upper middle income",
    "Cuba":                             "Upper middle income",
    "Curaçao":                          "High income",
    "Dominica":                         "Upper middle income",
    "Dominican Rep.":                   "Upper middle income",
    "Ecuador":                          "Upper middle income",
    "El Salvador":                      "Lower middle income",
    "Falkland Isds (Malvinas)":         "High income",
    "Greenland":                        "High income",
    "Grenada":                          "Upper middle income",
    "Guatemala":                        "Upper middle income",
    "Guyana":                           "Upper middle income",
    "Haiti":                            "Lower middle income",
    "Honduras":                         "Lower middle income",
    "Jamaica":                          "Upper middle income",
    "Mexico":                           "Upper middle income",
    "Montserrat":                       "Upper middle income",
    "Nicaragua":                        "Lower middle income",
    "Panama":                           "Upper middle income",
    "Paraguay":                         "Upper middle income",
    "Peru":                             "Upper middle income",
    "Saint Barthélemy":                 "High income",
    "Saint Kitts and Nevis":            "High income",
    "Saint Lucia":                      "Upper middle income",
    "Saint Maarten":                    "High income",
    "Saint Pierre and Miquelon":        "High income",
    "Saint Vincent and the Grenadines": "Upper middle income",
    "Suriname":                         "Upper middle income",
    "Trinidad and Tobago":              "High income",
    "Turks and Caicos Isds":            "High income",
    "Uruguay":                          "High income",
    "USA":                              "High income",
    "Venezuela":                        "Upper middle income",
    # ── Oceania (mapped to Americas continent) ────────────────────────────────
    "Australia":                        "High income",
    "Christmas Isds":                   "High income",
    "Cocos Isds":                       "High income",
    "Cook Isds":                        "Upper middle income",
    "Fiji":                             "Upper middle income",
    "FS Micronesia":                    "Lower middle income",
    "French Polynesia":                 "High income",
    "Guam":                             "High income",
    "Kiribati":                         "Lower middle income",
    "Marshall Isds":                    "Upper middle income",
    "Nauru":                            "High income",
    "New Caledonia":                    "High income",
    "New Zealand":                      "High income",
    "Niue":                             "Upper middle income",
    "Norfolk Isds":                     "High income",
    "Palau":                            "High income",
    "Papua New Guinea":                 "Lower middle income",
    "Pitcairn":                         "High income",
    "Samoa":                            "Lower middle income",
    "Solomon Isds":                     "Lower middle income",
    "Tokelau":                          "Upper middle income",
    "Tonga":                            "Lower middle income",
    "Tuvalu":                           "Lower middle income",
    "Vanuatu":                          "Lower middle income",
    "Wallis and Futuna Isds":           "High income",
}


# ── Parameter defaults (mirrors mass_balance.DEFAULTS; no circular import) ───

_GLOBAL_DEFAULTS: dict = {
    "gamma": 0.70, "tau": 3, "beta": 0.85, "delta": 0.95,
    "eta_break_formal": 0.95, "eta_break_informal": 0.78,
    "eta_ore": 0.95,
    "eta_scrap_formal": 0.97, "eta_scrap_informal": 0.70,
    "eta_mfg": 0.98, "eta_refine": 1.00,
    "phi_break_informal": 0.0, "phi_smelt_informal": 0.0,
    "lambda_divert": 0.05,
    "informal_mode": False,
}


def get_country_params(
    baci_country: str,
    params_df: pd.DataFrame,
    continent_df: pd.DataFrame,
) -> tuple[dict, bool, str]:
    """
    Return (params_dict, is_fallback, continent_name) for a country.

    Lookup order:
      1. country_parameters.csv (matched on BACI_name)
      2. continent_defaults.csv (matched on Continent + income_group, then Continent only)
      3. Global defaults

    The returned params_dict contains all keys needed by the V4 equation functions,
    including V4-specific fields: phi_break, phi_smelt, lambda_divert, informal_mode,
    income_group, continent, peer_country, param_source.
    """
    continent    = COUNTRY_CONTINENT_MAP.get(baci_country, "Africa")
    income_group = COUNTRY_INCOME_GROUP.get(baci_country, "Lower middle income")

    match = params_df[params_df["BACI_name"] == baci_country]
    if not match.empty:
        row = match.iloc[0]
        _f = lambda col, default: float(row[col]) if pd.notna(row.get(col)) else default
        _i = lambda col, default: int(row[col])   if pd.notna(row.get(col)) else default
        _b = lambda col, default: bool(row[col])  if pd.notna(row.get(col)) else default
        _s = lambda col, default: str(row[col])   if pd.notna(row.get(col)) else default
        p = {
            # Core parameters
            "gamma":               _f("gamma",               _GLOBAL_DEFAULTS["gamma"]),
            "tau":                 _i("tau",                  _GLOBAL_DEFAULTS["tau"]),
            "beta":                _f("beta",                 _GLOBAL_DEFAULTS["beta"]),
            "delta":               _GLOBAL_DEFAULTS["delta"],
            # Efficiency parameters — formal and informal streams
            "eta_break_formal":    _f("eta_break_formal",    _GLOBAL_DEFAULTS["eta_break_formal"]),
            "eta_break_informal":  _f("eta_break_informal",  _GLOBAL_DEFAULTS["eta_break_informal"]),
            "eta_ore":             _GLOBAL_DEFAULTS["eta_ore"],
            "eta_scrap_formal":    _f("eta_scrap_formal",    _GLOBAL_DEFAULTS["eta_scrap_formal"]),
            "eta_scrap_informal":  _f("eta_scrap_informal",  _GLOBAL_DEFAULTS["eta_scrap_informal"]),
            "eta_mfg":             _f("eta_mfg",              _GLOBAL_DEFAULTS["eta_mfg"]),
            "eta_refine":          _GLOBAL_DEFAULTS["eta_refine"],
            # V4 split parameters (active only in Informal Mode)
            # Both represent INFORMAL share; both default to 0
            "phi_break_informal":  _f("phi_break_default",   _GLOBAL_DEFAULTS["phi_break_informal"]),
            "phi_smelt_informal":  _f("phi_smelt_default",   _GLOBAL_DEFAULTS["phi_smelt_informal"]),
            "lambda_divert":       _f("lambda_divert",        _GLOBAL_DEFAULTS["lambda_divert"]),
            "informal_mode":       _b("informal_mode",        False),
            # Metadata
            "income_group":        _s("income_group",         income_group),
            "continent":           _s("continent",            continent),
            "peer_country":        _s("peer_country",         ""),
            "param_source":        "country_specific",
            # Convenience fields for UI sliders
            "SLI_pct":             _i("SLI_pct",              60),
            # Mining override (kt → t); None means use mining_production.csv
            "mining_t_override":   (
                float(row["mining_kt"]) * 1000.0
                if pd.notna(row.get("mining_kt")) and float(row.get("mining_kt", 0) or 0) > 0
                else None
            ),
        }
        # Backward-compat aliases used by older UI code
        p["eta_break"] = p["eta_break_formal"]
        p["eta_scrap"] = p["eta_scrap_formal"]
        return p, False, continent

    # ── Continent + income_group fallback ────────────────────────────────────
    cont_row = continent_df[
        (continent_df["Continent"] == continent) &
        (continent_df["income_group"] == income_group)
    ]
    if cont_row.empty:
        cont_row = continent_df[continent_df["Continent"] == continent]
    if not cont_row.empty:
        r = cont_row.iloc[0]
        _cf = lambda col, default: float(r[col]) if pd.notna(r.get(col)) else default
        _ci = lambda col, default: int(r[col])   if pd.notna(r.get(col)) else default
        p = {
            "gamma":               _cf("gamma",              _GLOBAL_DEFAULTS["gamma"]),
            "tau":                 _ci("tau",                 _GLOBAL_DEFAULTS["tau"]),
            "beta":                _cf("beta",                _GLOBAL_DEFAULTS["beta"]),
            "delta":               _GLOBAL_DEFAULTS["delta"],
            "eta_break_formal":    _cf("eta_break_formal",   _GLOBAL_DEFAULTS["eta_break_formal"]),
            "eta_break_informal":  _cf("eta_break_informal", _GLOBAL_DEFAULTS["eta_break_informal"]),
            "eta_ore":             _GLOBAL_DEFAULTS["eta_ore"],
            "eta_scrap_formal":    _cf("eta_scrap_formal",   _GLOBAL_DEFAULTS["eta_scrap_formal"]),
            "eta_scrap_informal":  _cf("eta_scrap_informal", _GLOBAL_DEFAULTS["eta_scrap_informal"]),
            "eta_mfg":             _cf("eta_mfg",             _GLOBAL_DEFAULTS["eta_mfg"]),
            "eta_refine":          _GLOBAL_DEFAULTS["eta_refine"],
            "phi_break_informal":  _GLOBAL_DEFAULTS["phi_break_informal"],
            "phi_smelt_informal":  _GLOBAL_DEFAULTS["phi_smelt_informal"],
            "lambda_divert":       _GLOBAL_DEFAULTS["lambda_divert"],
            "informal_mode":       False,
            "income_group":        income_group,
            "continent":           continent,
            "peer_country":        "",
            "param_source":        f"continent_fallback_{continent}",
            "SLI_pct":             _ci("SLI_pct", 60),
            "mining_t_override":   None,
        }
        p["eta_break"] = p["eta_break_formal"]
        p["eta_scrap"] = p["eta_scrap_formal"]
        return p, True, continent

    # ── Final fallback: global defaults ──────────────────────────────────────
    p = dict(_GLOBAL_DEFAULTS)
    p.update({
        "income_group":         income_group,
        "continent":            continent,
        "peer_country":         "",
        "param_source":         "global_default",
        "SLI_pct":              60,
        "mining_t_override":    None,
        # Backward-compat aliases
        "eta_break":            _GLOBAL_DEFAULTS["eta_break_formal"],
        "eta_scrap":            _GLOBAL_DEFAULTS["eta_scrap_formal"],
    })
    return p, True, continent
