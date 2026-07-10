# Pb Action Portal

The public Streamlit app for the **Partnership for Battery Action (Pb Action / GDI)** —
a simplified, no-password subset of the internal Pb Action Toolkit.

It presents **one flat level of tabs**:

| Tab | What it shows |
|-----|---------------|
| **Trade Map** | Bilateral BACI lead-trade flows on a world map |
| **Trade Trends** | Time series of imports/exports/production by country |
| **Trade Relationships** | Flow-network graph of who trades with whom |
| **Lead Accumulation** | Net lead balance (mining + imports − exports) over time |
| **Production & Capacity** | Where lead is mined, refined, and where batteries are made |
| **Recycling Economy Snapshot** | Country recycling-economy overview |
| **Material Flow (Beta)** | Experimental material-flow Sankey per country |
| **Literature Stats** | Searchable library of sourced datapoints |

## Easy vs. Advanced mode

A **View mode** toggle in the sidebar trades usability for control:

- **Easy** fixes sensible data assumptions and hides the detailed controls.
- **Advanced** exposes every input (dataset, sources, time period, Pb-content
  factors) so you can adjust the model and challenge its assumptions.

## Data & sources

The Portal draws together third-party datasets that the Pb Action team has
collected, cleaned, and harmonized. The datasets remain those of their original
publishers; the model outputs, estimates, and BOTEC-type calculations are the
project's own analysis. All values are expressed in tonnes of lead content.

| Data | Source |
|---|---|
| Bilateral trade flows | BACI / CEPII (harmonized UN Comtrade) |
| Mine & refined production | USGS, BGS |
| Battery collection | Eurostat, BCI, ILZSG, national studies |
| Literature figures | Published studies — per-figure citations on the **Literature Stats** tab |

The **Recycling Economy Snapshot** and **Material Flow (Beta)** tabs are
back-of-the-envelope (BOTEC) estimates, labeled as such in-app.

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

No password is required.

## Literature Stats submissions (deferred)

A form for visitors to submit new datapoints (writing to a Google Sheet for
review) is built but **disabled for now** — see `literature/submit.py` and the
commented `render_submission_form()` call in `literature/app.py`. To re-enable it
later: uncomment those two lines, restore `gspread` / `google-auth` in
`requirements.txt`, and configure the secrets (`.streamlit/secrets.toml.example`).

## Relationship to the Toolkit

This repo is generated from the private **Pb Action Toolkit**. The Toolkit is the
sandbox where features mature; the Portal is always a strict subset. Shared code
(`model/`, `visualizations/`, `literature/`) and the `data/` summaries are copied
from the Toolkit — edit those upstream, not here.
