# Lead Accumulation Tab — Methodology & Reference

## Overview

The **Lead Accumulation** tab estimates the annual net lead balance for any country
or sub-region covered by BACI trade data (2012–2024). The calculation answers:
*How much lead entered or left this economy in a given year, and through which channels?*

## Core Equation

```
Net(country, year) = Mining(year) + Imports_Pb(year) − Exports_Pb(year)
```

- **Positive net** → the country is a net recipient of lead (mining + imports exceed exports)
- **Negative net** → the country is a net exporter of lead

## Data Sources

| Source | File | What it provides |
|--------|------|-----------------|
| BACI trade data (master) | `data/BACI_lead_trade_2012_2024_modified_vHS_4_master.csv` | Bilateral trade flows 2012–2024 for all 17 lead HS codes |
| USGS mineral yearbooks | `data/mined.csv` | Annual mine production by country, already in tonnes Pb content |

The master BACI file covers a broader set of HS codes than the battery-focused file used
in other tabs. It does **not** have a pre-computed Pb Quantity column; factors are applied
in this tab using the values below.

## HS Code Categories and Pb Content Factors

| HS Code | Description | Category | Default Pb% | Adjustable range |
|---------|-------------|----------|------------|-----------------|
| 260700 | Lead Ore & Concentrates | Mining | 60% | 45–75% |
| 262021 | Leaded Gasoline Residues | Slag | 40% | 20–60% |
| 262029 | Other Lead-Containing Residues | Slag | 55% | 30–80% |
| 282410 | Lead Monoxide | Lead Feedstock | 93% | fixed |
| 282490 | Other Lead Oxides | Lead Feedstock | 90% | 87–93% |
| 780110 | Refined Unwrought Lead | Lead Feedstock | 99% | fixed |
| 780191 | Unwrought Lead (w/ Antimony) | Lead Feedstock | 96% | 94–98% |
| 780199 | Other Unwrought Lead | Lead Feedstock | 97% | 95–99% |
| 780200 | Lead Waste & Scrap | Waste Batteries & Scrap | 72.5% | 50–95% |
| 780411 | Lead Sheet/Strip/Foil (PCB) | Other Lead Products | 99% | fixed |
| 780419 | Lead Sheet/Strip/Foil (Other) | Other Lead Products | 99% | fixed |
| 780420 | Lead Tubes, Pipes & Fittings | Other Lead Products | 98.5% | 98–99% |
| 780600 | Other Lead Articles | Other Lead Products | 97% | 95–99% |
| 850710 | Lead-Acid Starter Batteries | New Batteries | 60% | 55–65% |
| 850720 | Other Lead-Acid Batteries | New Batteries | 60% | 55–65% |
| 850790 | Battery Parts | Other Lead Products | 62.5% | 30–95% |
| 854810 | Used Lead-Acid Batteries | Waste Batteries & Scrap | 60% | 55–65% |

Default values are midpoints of the stated range. BACI `Quantity` is in metric tonnes of
raw commodity; multiplied by the factor to get tonnes of lead content (t Pb).

BGS/USGS mine-production values are already in tonnes Pb — **do not re-apply a factor**.

## Category Definitions

| Category | Components |
|----------|-----------|
| **Mining** | BGS/USGS mine production + net trade in HS 260700 (ore & concentrates) |
| **Lead Feedstock** | HS 780110, 780191, 780199 (unwrought lead) + 282410, 282490 (oxides) |
| **New Batteries** | HS 850710 (SLI / starter) + 850720 (industrial) |
| **Waste Batteries & Scrap** | HS 854810 (used batteries) + 780200 (scrap) |
| **Slag** | HS 262021 (leaded gasoline residues) + 262029 (other lead-containing residues) |
| **Other Lead Products** | HS 780411, 780419, 780420, 780600 (fabricated lead) + 850790 (battery parts) |

## Sub-Region Treatment

When a sub-region is selected, **only external trade is counted** — flows between two
countries both inside the region are excluded to avoid double-counting:

```
Imports (region) = rows where Importer ∈ region AND Exporter ∉ region
Exports (region) = rows where Exporter ∈ region AND Importer ∉ region
Mining (region)  = sum of all countries in region
```

Sub-region definitions follow UN M49 geography (see `model/regions.py`).

## Country Name Alignment

BACI and USGS use different country naming conventions. The following mappings are
applied when joining mine-production data to BACI country names:

| USGS name | BACI name |
|-----------|-----------|
| Republic Of Korea | Rep. of Korea |
| Burma | Myanmar |
| North Korea | Dem. People's Rep. of Korea |
| Macedonia | North Macedonia |
| Russia | Russian Federation |
| Laos | Lao People's Dem. Rep. |
| Vietnam | Viet Nam |
| Bolivia | Bolivia (Plurinational State of) |

If a mining country's name is not in this mapping, it is passed through unchanged.
Countries not present in `mined.csv` simply show 0 for mine production.

## Visualizations

### Line Chart — Annual Net Balance

- X-axis: year (2012–2024)
- Y-axis: net lead balance (kt Pb)
- Green fill above zero, red fill below zero
- Orange diamond marks the year selected for detail

### Bar Chart — Category Breakdown

- One bar per category for the selected year
- Positive = net lead gain from that category; negative = net loss
- Categories use consistent colors across the tab

### Breakdown Table

- Rows: each HS code with data in the selected year, grouped by category
- Bold gray rows: category subtotals
- Net column color: darker green = larger gain; darker red = larger loss
- Units: kt Pb (kilotonnes of lead content)

## Restoring the India Tab

The India Lead Model tab is preserved in `streamlit_app.py` inside an `if False:` block
(lines ~1356–1720). To re-enable:

1. Add `"India Lead Model"` as a 7th entry in the `st.tabs([...])` call
2. Capture it as `tab_india`
3. Wrap the `if False:` block as `with tab_india:`
4. Remove the `if False:` wrapper

## Known Limitations

- BACI `Quantity` is in tonnes of raw commodity weight. For heterogeneous product
  categories (850790 Battery Parts, 780200 Scrap), the Pb factor range is wide;
  results are sensitive to the assumed factor.
- Mine production data in `mined.csv` covers only countries with USGS records.
  Countries without USGS data (e.g., most of Africa) show 0 for mining.
- BACI does not capture informal/unreported trade, so small-country balances should
  be interpreted cautiously.
- The annual net balance is not the same as stock accumulation — lead that entered
  in prior years may have been processed and re-exported.
