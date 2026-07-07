# Pb Action — Global Lead Exploratory Portal

A Streamlit dashboard developed by the **Partnership for Battery Action (Pb Action)**,
housed within the Global Development Incubator (GDI). It gives a comprehensive,
data-driven picture of how lead flows through the global lead-acid battery
ecosystem — from mining and manufacturing through installation, collection,
breaking, and secondary smelting.

## What this is

The portal is the next evolution of the lead trade dashboard, combining a refined
version of the BACI bilateral trade dataset with new data sources and a novel
analytical framework. Where a trade dashboard shows *what crossed borders*, this
tool estimates *what happened inside countries* — and how each country fits into
the broader global lead ecosystem.

It brings together:

- **Trade data** — BACI / UN COMTRADE bilateral flows across 11 lead-related HS
  codes (2012–2024), with conversion factors applied so all quantities are
  expressed in tonnes of lead content.
- **Mining data** — Country-level mine production estimates from USGS and BGS.
- **Vehicle and fleet data** — Vehicles per 1,000 people by country, used to
  anchor battery installation estimates.
- **Collection rates** — Country-specific estimates of the share of end-of-life
  batteries formally collected (BCI, ILZSG, national studies, regional defaults).
- **Smelter and facility data** — To be integrated as Battery Index fieldwork matures.

## What's inside

The portal includes several analytical tools, each a tab:

- **Country-Level Lead Usage Model** — estimates how lead flows through each stage
  of the recycling loop for a single country (installation, collection, breaking,
  secondary smelting, manufacturing). Key parameters (collection rate, battery
  lifespan, recovery efficiencies) are adjustable in real time. This is the core
  analytical engine.
- **Country Comparison** — places multiple countries side by side on a radar chart,
  with a diagnostics panel that flags imbalances between breaking and smelting,
  gaps between modeled and observed installation, unexplained feedstock flows, and
  implied battery shares that deviate from known values.
- **Estimate Generator** — aggregates key lead statistics across any combination of
  countries, UN regions, and World Bank income groups.
- **Trade Flow Maps** — a bilateral country trade map (top partners for a selected
  product category) and a regional trade view treating UN regions as units.
- **Supply Chain Sankey** — traces lead through the recycling chain for a single
  country, from collection and imported waste batteries through breaking and
  secondary smelting to refined-lead exports and domestic manufacturing.
- **Supply Chain Provenance** — maps the full upstream supply chain behind a
  selected end-market country's battery supply.

## The model

The mass balance model at the core of the tool is a process-based, country-level
material flow analysis (MFA). It tracks lead by mass (tonnes of lead content)
through five stages of the battery lifecycle: ore processing, smelting,
manufacturing, installation, and collection/recycling. It estimates each stage
sequentially, using trade data as the primary observable and country-specific
parameters to fill gaps. It currently covers 123 countries with individually
calibrated parameters, with income-group and continent composite defaults for the
remainder.

See [`CLAUDE.md`](CLAUDE.md) for the full equation set, parameter definitions,
conversion factors, and diagnostics.

## Running locally

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The app then opens at http://localhost:8501. All data it needs ships in this
repository under `data/` and `india_model/` — no external downloads required.

### Repository layout

```
streamlit_app.py        Main entry point (tabs)
model/                  Data loading, region maps, shared constants
visualizations/         One module per tab (maps, radar, Sankey, flow network)
india_model/            India mass-balance model modules + input data
data/                   Trade, mining, collection, and reference datasets
docs/                   Supplementary notes
```

## Current status and known limitations

This portal is a working prototype under active development:

- Installation estimates are a lower bound for most LMICs; the current fleet-based
  method captures SLI batteries only. Industrial battery demand (UPS, telecom,
  off-grid storage, electric three-wheelers) is significant in lower-income
  countries but not yet reliably estimated.
- Collection rates for many countries rely on income-group defaults rather than
  country-specific data.
- Informal-sector flows are not directly observable; the model estimates them as
  residuals but cannot verify them.
- BACI data lags roughly 18 months; 2023–2024 figures should be treated as preliminary.
- Smelter-level data is not yet integrated; Battery Index fieldwork will eventually
  anchor facility-level throughput.
- Some UI elements are still under development.

The model is not designed for precise tonnage — it characterizes market scale and
structure and flags where recycling may be unsafe.

## Data sources

| Dataset | Source | Coverage |
|---|---|---|
| Bilateral trade flows | BACI / CEPII | 2012–2024, 11 HS codes |
| Mine production | USGS, BGS | 58 countries |
| Vehicle fleet | World Bank WDI, OICA, IRF | 123 countries |
| Battery share of lead demand (β) | ILZSG World Lead Factbook 2023 | Key economies |
| Collection rates | BCI, ILZSG, national studies, regional defaults | 123 countries |
| Installation anchors | USGS, Pahle India Foundation, SRI Policy Brief 2024 | 7 countries |

## Roadmap

- Complete calibration of installation estimates against ILZSG country-level
  consumption data.
- Integrate Battery Index facility-level smelter throughput.
- Expand the company database and connect it to country-level flows.
- Improve informal-sector estimation methodology.

## License

See [`LICENSE`](LICENSE).

## Contact

For questions, access requests, or to schedule a walkthrough, contact Ben Savonen
at ben.savonen@globaldevincubator.org.

<!-- sync test 19:38:39 -->
