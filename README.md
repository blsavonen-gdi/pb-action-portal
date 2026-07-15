# Pb Action Portal

The public Streamlit dashboard for the **Partnership for Battery Action
(Pb Action / GDI)** — a set of interactive tools for exploring how lead moves
through the global economy: international trade, mining and refining, battery
manufacturing, and end-of-life recycling.

All quantities are expressed in **metric tonnes of lead (Pb) content** (the model
tabs display kilotonnes). The Portal is built for high-level analysis and pattern
detection; results should be cross-checked against primary sources before drawing
conclusions. There is no password — the only interstitial is a one-time
data-quality disclaimer.

> **Full methodology.** This README is a summary. The complete tab-by-tab
> methodology and user guide is in [`docs/Portal_README.pdf`](docs/Portal_README.pdf)
> (source: [`docs/Portal_README.tex`](docs/Portal_README.tex)).

## Tabs

One flat level of eight tabs:

| Tab | What it shows |
|-----|---------------|
| **Trade Map** | Bilateral lead-trade flows on a world map (net position, bilateral partners, or regional external flows) |
| **Trade Trends** | Import / export / production time series by country |
| **Trade Relationships** | Flow-network graph of who trades with whom |
| **Lead Accumulation** | Net lead balance (mining + imports − exports) over time |
| **Production & Capacity** | Where lead is mined, refined, and where batteries are made |
| **Recycling Economy Snapshot (Beta)** | Radar comparison of countries' recycling economies |
| **Material Flow (Beta)** | Per-country material-flow Sankey |
| **Literature Stats** | Searchable library of sourced datapoints |

## Easy vs. Advanced mode

A **View mode** toggle in the sidebar trades usability for control:

- **Easy** (default) fixes sensible data assumptions and hides the detailed
  controls — always HS12, BGS sources, a 3-year centered average, and default
  Pb-content factors (Slag and Other Lead Products excluded). It exposes only a
  center-year slider.
- **Advanced** exposes every input: dataset, sources, time period, per-HS
  Pb-content factors, and extra per-tab controls (color modes, layouts, animation,
  process-parameter sliders).

## How the numbers are built

- **Datasets.** BACI (CEPII, harmonized UN Comtrade). *HS12 (2012–2024)* uses HS
  2017 codes with waste-battery code `854810`; *HS22 (2022–2024)* uses HS 2022 codes
  with the reclassified `854911`. (In the pre-2022 history, `854810` is used as a
  proxy for spent lead-acid batteries; it is a defensible proxy but biases those
  volumes somewhat high near 2022 — see the methodology PDF footnote.)
- **Mining & refining sources.** BGS (1971–2023) or USGS (2015–2023, primary/secondary
  split), with automatic fallback when a country/year is missing.
- **Time period.** Single year, or a 3-year average (recommended) annualised by
  summing flows and dividing by the number of years.
- **Pb-content factors.** Each HS code is multiplied by a lead-content fraction.
  Adjustable in Advanced mode (Slag and Other Lead Products off by default). The
  trade tabs read a pre-computed lead-content column; the model tabs re-derive lead
  live from the adjustable factors.
- **Lead Accumulation** = mining + imports − exports, cumulative since 2012.

## The material-flow (BOTEC) engine

The Material Flow Sankey, Recycling Economy Snapshot, and the two Production BOTEC
datasets share one engine. It anchors on a country's reported secondary
(scrap-based) smelting output and solves a **backward chain** (smelting → breaking
→ collection) and a **forward chain** (feedstock → non-battery/battery →
manufacturing → installed batteries). Default process parameters (all adjustable in
Advanced): secondary smelting 0.97, breaking 0.95, end-of-life retention 0.95,
battery share 0.85, manufacturing 0.98, primary smelting 0.95, refining 0.97, and
**collection rate 0.95** (consistent across all material-flow views).

These are back-of-the-envelope estimates: a single deterministic pass with global
default parameters, not a calibrated fit. Collection and recovery rates vary
enormously by country (collection alone runs ~0.99 in the USA/Japan, ~0.70 in India,
~0.60 in Nigeria), so **switch to Advanced mode and tune the parameters for the
country and year you are studying** — the defaults are an illustrative starting
point, not a country-specific answer.

## Data & sources

| Data | Source |
|---|---|
| Bilateral trade flows | BACI / CEPII (harmonized UN Comtrade) |
| Mine & refined production | USGS, BGS |
| Battery collection | Eurostat, BCI, ILZSG, national studies |
| Literature figures | Published studies — per-figure citations on the **Literature Stats** tab |

The datasets remain those of their original publishers; the model outputs,
estimates, and BOTEC calculations are the project's own analysis.

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

No password is required. All data the app needs ships in this repo. The app is
deployed on Streamlit Community Cloud and auto-redeploys on push to `main`.

## Literature Stats submissions (deferred)

A form for visitors to submit new datapoints (writing to a Google Sheet for review)
is built but **disabled for now** — see `literature/submit.py` and the commented
`render_submission_form()` call in `literature/app.py`. To re-enable it: uncomment
those lines, restore `gspread` / `google-auth` in `requirements.txt`, and configure
the secrets (`.streamlit/secrets.toml.example`).

## License

See [`LICENSE`](LICENSE).
