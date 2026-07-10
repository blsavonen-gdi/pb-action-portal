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
| **Literature Stats** | Searchable library of sourced datapoints + a submission form |

## Easy vs. Advanced mode

A **View mode** toggle in the sidebar trades usability for control:

- **Easy** fixes sensible data assumptions and hides the detailed controls.
- **Advanced** exposes every input (dataset, sources, time period, Pb-content
  factors) so you can adjust the model and challenge its assumptions.

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

No password is required.

## Literature Stats submissions → Google Sheet

The **Submit a new statistic** form on the Literature Stats tab appends each
submission to a Google Sheet for review.

1. Create a Google Cloud service account + JSON key; enable the **Google Sheets API**.
2. Create a Google Sheet and share it (Editor) with the service account email.
3. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill in
   the key + sheet id (locally), or paste the same into **App settings → Secrets**
   on Streamlit Community Cloud.

If the secret isn't configured, the form still works: on submit it offers the
entry as a CSV download plus a pre-filled email link, so nothing is lost.

## Relationship to the Toolkit

This repo is generated from the private **Pb Action Toolkit**. The Toolkit is the
sandbox where features mature; the Portal is always a strict subset. Shared code
(`model/`, `visualizations/`, `literature/`) and the `data/` summaries are copied
from the Toolkit — edit those upstream, not here.
