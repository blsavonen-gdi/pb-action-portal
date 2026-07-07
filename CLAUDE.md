# CLAUDE.md — Pb Action Mass Balance Model

## What This Project Is

This is a Streamlit web app for the **Partnership for Battery Action (Pb Action)**, housed within the Global Development Incubator (GDI). It implements a country-level lead mass balance model that estimates how much lead-acid battery recycling is happening in each country — and flags where that recycling may be unsafe.

The app should allow users to explore the model interactively: select countries, adjust parameters, see radar plots of the recycling loop, compare countries, view time series, and examine diagnostic flags.

## Who Uses This

Internal team (Pb Action analysts, researchers, partners), funders (Coefficient Giving, World Bank), and potentially external stakeholders (UNEP, government officials). The audience is quantitatively literate but not necessarily technical — the app should explain what it's showing.

## Core Data Files

All in the `data/` directory:

| File | Description | Encoding |
|------|-------------|----------|
| `BACI_lead_trade_2012_2024_modified_vHS_4.csv` | Main trade dataset. Covers 2012–2024, all 11 HS codes including 850790, 282410, 282490. 8 columns: Year, Exporter, Importer, Product, Value, Quantity, Pb Quantity. The `Pb Quantity` column already has conversion factors applied — do not multiply by HS code conversion factors again. | `utf-8-sig` |
| `installation_estimates.csv` | 44 countries, columns: Country, Lead in Batteries Installed (kt Pb/yr), Year, Method, Sources | default |
| `mining_production.csv` | 58 countries, columns: Country, Mining Production (tonnes Pb content), Year, Source | default |
| `country_parameters.csv` | Per-country parameter starting estimates (see below) | default |

### BACI Column Notes
- `Pb Quantity` is already in tonnes of lead content — conversion factors pre-applied
- `Quantity` is raw trade quantity (do not use for lead calculations)
- `Product` is the HS code as integer (260700, 780110, etc.)
- Country names use BACI conventions (e.g., 'Rep. of Korea', 'USA', 'Côte d'Ivoire')
- Türkiye encoding issue handled in data_loader.py via BACI_TO_STANDARD_NAME mapping

## The Model (V3)

### Conceptual Framework
Lead-acid batteries form a closed loop: Mining → Smelting → Manufacturing → Installation → Collection → Breaking → Smelting. The model tracks lead (by mass) through this loop for each country.

Five product categories map to five processing stages:
- **ORE** (260700) → Mining
- **FEED** (780110, 780191, 780199, 850790, 282410, 282490) → Smelting output
- **BATT** (850710, 850720) → Manufacturing output
- **USED** (854810) → Collection output
- **SCRAP** (780200) → Breaking output

### Lead Content Conversion Factors
Applied upstream to raw BACI data (the `Actual Lead` column). If users want to adjust these in the app, apply them to the `quantity` column instead.

| HS Code | Product | Factor | Source |
|---------|---------|--------|--------|
| 260700 | Ore & concentrates | 0.60 | Trade-weighted avg (Britannica) |
| 780110 | Refined lead | 1.00 | 99%+ purity by definition |
| 780191 | Antimonial lead | 0.95 | 1-6% Sb typical (ILZSG) |
| 780199 | Other unwrought | 0.95 | Minor alloying (ILZSG) |
| 780200 | Scrap | 0.97 | Metallic Pb scrap |
| 850710 | SLI batteries | 0.65 | CARE Ratings, battery teardowns |
| 850720 | Industrial batteries | 0.70 | Thicker plates, less casing |
| 850790 | Battery parts | 0.80 | Mostly plates/grids |
| 854810 | Waste batteries | 0.70 | Acid typically drained |
| 282410 | Lead oxides | 0.91 | Stoichiometric (PbO=92.8%) |
| 282490 | Other compounds | 0.75 | Heterogeneous; central estimate |

### Equations (solved sequentially)

All quantities in metric tonnes of lead content. IMP/EXP from BACI. M from mining data. INSTALL from installation estimates.

```
Eq 1:  ORE_DOM(i,t)      = M(i,t) + IMP_ore(i,t) - EXP_ore(i,t)
Eq 2:  COLLECT(i,t)       = γ(i) × INSTALL(i, t-τ)
Eq 3:  DISPOSAL(i,t)      = (1 - γ(i)) × INSTALL(i, t-τ)
Eq 4:  USED_DOM(i,t)      = COLLECT(i,t) + IMP_used(i,t) + W_unrec(i,t) - EXP_used(i,t)
Eq 5:  BREAK(i,t)         = max(0, USED_DOM(i,t)) × δ × η_break
Eq 6:  SCRAP_DOM(i,t)     = BREAK(i,t) + IMP_scrap(i,t) - EXP_scrap(i,t)
Eq 7:  SMELT_PRIMARY(i,t) = max(0, ORE_DOM(i,t)) × η_ore
Eq 8:  SMELT_SECONDARY(i,t) = max(0, SCRAP_DOM(i,t)) × η_scrap
Eq 9:  SMELT(i,t)         = SMELT_PRIMARY(i,t) + SMELT_SECONDARY(i,t)
Eq 10: FEED_DOM(i,t)      = SMELT(i,t) + IMP_feed(i,t) - EXP_feed(i,t)
Eq 11: NON_BATT(i,t)      = FEED_DOM(i,t) × (1 - β)
Eq 12: FEED_BATT(i,t)     = FEED_DOM(i,t) × β × η_mfg
Eq 13: MFG(i,t)           = max(0, FEED_BATT(i,t))
Eq 14: INSTALL_IMPLIED(i,t) = MFG(i,t) + IMP_batt(i,t) - EXP_batt(i,t)
```

### Parameters

| Symbol | Description | Default | Notes |
|--------|-------------|---------|-------|
| β | Battery share of Pb demand | 0.85 | JP:0.97, CN/US:0.92, IN:0.75 |
| γ(i) | Collection rate | 0.70 | Country-specific (see CSV) |
| τ | Battery lifespan lag (years) | 3 | Can derive from 850710/850720 import ratio |
| δ | Pb remaining at end-of-life | 0.95 | Plate degradation |
| η_break | Breaking recovery | 0.95 | Formal; informal may be 0.70-0.85 |
| η_ore | Primary smelting recovery | 0.95 | |
| η_scrap | Secondary smelting recovery | 0.97 | Formal; informal may be 0.60-0.80 |
| η_mfg | Manufacturing efficiency | 0.98 | |
| W_unrec(i,t) | Unrecorded waste imports | 0 | Ghana: ~5000 t/yr |

### Deriving τ from trade data
The ratio of 850710 (SLI) to 850720 (industrial) battery imports tells you the auto/industrial mix. SLI batteries last ~3 years (tropical) to ~5 years (cold climate). Industrial batteries last ~5-8 years. Weighted τ = (SLI_share × τ_auto) + (IND_share × τ_industrial).

### Country-Specific Collection Rates

| Country | γ | Basis |
|---------|---|-------|
| USA | 0.99 | BCI data |
| Japan | 0.98 | Industry reports |
| S. Korea, Germany | 0.95 | High-income formal systems |
| Ghana | 0.85 | Calibrated to observed exports |
| China | 0.85 | Mixed formal/informal |
| Thailand | 0.80 | |
| Brazil | 0.75 | |
| India | 0.70 | Large informal sector |
| Nigeria | 0.60 | Lower formal infrastructure |
| Default | 0.70 | Applied where no data |

### Diagnostics

**D2: Break-to-Smelt Ratio** = BREAK / SMELT_SECONDARY
- ~1.0 = balanced. >1.3 = excess scrap (exported or informally processed). <0.85 = smelters importing scrap.

**D3: Installation Gap** = INSTALL_exogenous - INSTALL_implied
- Positive = closed domestic loop (invisible to trade). Negative = install estimate too low or β too high.

**D5: Feedstock Coverage Ratio** (3-year rolling) = Σ SMELT_SEC / Σ FEED_EXPORTS
- FOR NET FEED EXPORTERS ONLY. 1.0 = fully explained. <1.0 = persistent untracked feedstock.
- Uses 3-year window to smooth stockpile effects.
- >1.0 expected for integrated economies (most smelting consumed domestically).

**D4 was dropped** — algebraically redundant when smelting is derived from scrap balance. Would become meaningful with independent smelting data (e.g., Battery Index facility throughput).

## Key Findings to Reproduce in the App

### Country Typologies
- **Export processors** (Ghana, Nigeria): Smelting/breaking spike, manufacturing ≈ 0. Broken loop.
- **Integrated exporters** (S. Korea, Japan, China): All stages active, manufacturing > installation (net battery exporters).
- **Domestic loops** (USA, Germany, Thailand, Brazil): Mostly self-contained recycling.
- **Import-dependent** (most LMICs): Consume batteries, don't manufacture or smelt.

### Ghana Time Series (2012-2024)
- **Pre-2020**: Transit hub. Ghana collected and EXPORTED waste batteries (to Togo, India). Domestic smelting near zero from recycling loop. Refined lead exports (4-12 kt/yr) came from non-battery feedstock.
- **2020 pivot**: Gravita Ghana and others scaled up. Waste battery exports dropped to zero, imports surged.
- **2021-2024**: Model converges to 77-94% of observed exports. Persistent ~23% gap (cumulative 2021-2023) = ~4,600 t/yr untracked feedstock.

### West Africa Regional Analysis (2020-2023)
- **Transit hubs**: Togo (exports 8x its domestic collection), Burkina Faso, Mali, Guinea, Sierra Leone
- **Processors**: Ghana (net importer of waste), Senegal and Cameroon also smelt domestically
- **Key destination mapping**: Burkina→Ghana (78%), Mali→Ghana (100%), Togo→India (64%) and Korea (29%), Guinea→India (84%)
- **Regional model**: Even treating all of West Africa as one country, a 20% gap persists between modeled smelting and observed exports

### Nigeria
- Model explains only 25-47% of observed exports (D5=0.25 on 3-year rolling)
- Trade-derived installation (25,750 t) is likely 3-5x undercounted
- ULAB generation estimates: Low 69kt, Mid 137kt, High 206kt. Low estimate closes most of the gap.

## Visualizations to Build

### 1. Recycling Loop Radar (highest priority)
Five axes in this order: Manufacturing → Installation → Collection → Breaking → Secondary Smelting.
Dashed pentagon at installation level = "circular economy baseline" (100%).
Toggleable countries. Normalized to each country's installation.
Hover shows absolute tonnage.
We have a working HTML version: `lead_recycling_loop_radar_v3.html`

### 2. Time Series (for countries with multi-year data)
Model-estimated smelting vs observed feed exports by year.
Cumulative gap analysis panel.
Waste battery import/export bar chart showing the transit hub → processor transition.

### 3. Diagnostic Dashboard
Table showing D2, D3, D5 for all countries.
Flag interpretation (color-coded or icon-based).
Country type classification.

### 4. Regional View (West Africa)
Waste battery flow network (who feeds whom).
Regional mass balance treating the region as one unit.

### 5. Parameter Adjustment Panel
Sliders for all adjustable parameters (γ, τ, β, δ, η values).
Conversion factors adjustable.
Real-time recalculation of model outputs.

## App Structure Suggestion

```
app/
├── streamlit_app.py          # Main entry point
├── model/
│   ├── mass_balance.py       # Core model equations
│   ├── diagnostics.py        # D2, D3, D5 calculations
│   └── data_loader.py        # Load and preprocess CSVs
├── visualizations/
│   ├── radar.py              # Recycling loop radar chart
│   ├── timeseries.py         # Time series panels
│   ├── diagnostics_table.py  # Diagnostic dashboard
│   └── regional.py           # West Africa regional view
├── data/
│   ├── BACI_lead_trade_2012_2024_modified_vHS_4.csv
│   ├── installation_estimates.csv
│   ├── mining_production.csv
│   └── country_parameters.csv
├── docs/
│   ├── mass_balance_model.pdf        # Equations document
│   ├── model_decisions_log.pdf       # Assumptions & decisions
│   └── methodology_and_context.pdf   # Full methodology paper
├── CLAUDE.md                 # This file
├── README.md
└── requirements.txt          # streamlit, pandas, plotly, etc.
```

## Formatting & UX Preferences

- Ben prefers clean, professional interfaces. Not flashy.
- Blue input cells / formula-driven calculations pattern from Excel background.
- Parameter adjustments should feel immediate (Streamlit sliders with real-time recalc).
- Tables should be sortable and downloadable.
- Charts should be interactive (Plotly preferred over static matplotlib).
- The app should work without explanation — labels, tooltips, and inline notes rather than separate documentation pages.

## Technical Notes

- BACI CSV requires `encoding='utf-8-sig'` in pandas
- All mass balance quantities in metric tonnes of lead content
- `max(0, ...)` clamping on ORE_DOM and SCRAP_DOM before applying recovery rates
- Sequential solving (not simultaneous) — τ-year lag breaks circularity
- For countries without domestic manufacturing: INSTALL ≈ net battery imports from BACI
- For countries with large domestic loops: INSTALL from exogenous estimates (installation_estimates.csv)

## Comparable Models (for context, not for implementation)

This model sits in the tradition of material flow analysis (MFA). Comparable approaches:
- Mao, Dong & Graedel (2008): Lead MFA, 52 countries, ~20 params/country
- Dong et al. (2025): China dynamic MFA, Weibull lifetimes, 12+ efficiencies
- Sea Around Us: Fisheries catch reconstruction, 270+ countries, 15-30 params/country
- Trase: Deforestation supply chain mapping, SEI-PCS methodology
- GHG Protocol Scope 3: 30-100+ emission factors per company

Our model at 7 core parameters + 11 conversion factors is at the simpler end of this range, which is appropriate for our purpose (market characterization and risk flagging, not precise tonnage).

## What This Model Is NOT

- It cannot distinguish primary from secondary refined lead in trade data (HS 780110 covers both)
- It cannot tell you whether a specific smelter is operating safely (that's the Battery Index's job)
- It cannot see perfectly circular domestic economies (no trade signal)
- It is not designed for precise tonnage — it characterizes market scale and structure
- It does not cover China as an intervention target (out of scope for Pb Action's strategy)

## Relationship to trade.leadbatteries.org

Hugo Smith's tool at trade.leadbatteries.org is a BACI trade data explorer — it shows what crossed borders. Our model estimates what happened inside the country. The trade explorer can tell you Ghana exported 25,000 t of refined lead. Our model tells you that ~24,000 t came from secondary smelting fed by domestic collection plus imported waste batteries, and ~12% of the feedstock is untracked. Different tools, complementary purposes.

## Publishing updates (public repo)

This repo (`PbAction`) is the **private working repo** — it keeps the full
research history, calibration sweeps, Monte-Carlo runs, and scraped data, and is
not shared. The **public** app lives at
[`blsavonen-gdi/pb-action-portal`](https://github.com/blsavonen-gdi/pb-action-portal),
checked out locally at `../PbAction-public`. It contains only the ~44 files the
running Streamlit app needs (~40 MB) and is the source Streamlit Community Cloud
deploys from.

To release changes after editing the app here:

```bash
python scripts/build_public_repo.py ../PbAction-public --push
```

That script (`scripts/build_public_repo.py`) copies the keep-set (defined by an
explicit manifest at the top of the file), prunes the orphan `regional_map.py`,
excludes research scratch, then commits and pushes the public repo. Streamlit
Cloud auto-redeploys on push to `main`. Omit `--push` to sync files only.

**When you add a new tab, module, or data file to the app, add it to the manifest
(`FILES` / `DIRS`) in `scripts/build_public_repo.py`** — otherwise it won't reach
the public repo and the deployed app will break on a missing import/file. The
script prints a warning and exits non-zero if any manifest entry is missing.
