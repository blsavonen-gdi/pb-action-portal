"""India lead mass balance — model v4 (Pass B, revised).

Pure-function rebuild of the calibration chain. No I/O inside math functions,
no optimization. Thin loaders (load_segments, load_inputs) are provided as
conveniences.

Three-stage chain: BREAK -> SMELT (crude) -> REFINE.
Each of break and smelt has its own formal/informal split. Refine is fully
formal. HS 780199 (unrefined / "other unwrought") attaches BETWEEN smelt and
refine (CRUDE_supply). HS 780191 stays in FEED as refined.

Surface:
  B1 — retirement engine
    load_segments, tau_eff, fit_growth_rate, retire_rate, retire_flow

  B2 — three-stage forward chain
    load_inputs(path, smooth_window=None, trim_years=None)
    forward_chain(arr, k_stock, phi_break_f, phi_smelt_f, *,
                  g, tau, beta=0.75,
                  delta, eta_break_F, eta_break_I,
                  eta_smelt_F, eta_smelt_I, eta_refine, eta_mfg)

Structural rules (enforced by this module):
  - gamma TOTAL collection rate fixed at 0.98 (Dalberg/USAID). Split happens
    downstream at break/smelt, so gamma is the *total* collection rate.
  - beta FIXED at 0.75 (ILZSG). Not solved. 0.90 available as sensitivity.
  - tau is the harmonic-mean effective lifetime from segment config; g is
    fit from the raw stock series; RETIRE = eff_stock * g/(exp(g*tau)-1).
  - phi_break_f < phi_smelt_f < 1 (structural ordering; break is the MOST
    informal stage, smelt is less informal).
  - phi_refine = 1 (fully formal; refining is the bottleneck everything
    funnels through).
  - USGS *secondary* refined enters ONLY res_refine. Never injected into
    the forward chain. USGS *primary* refined enters as exogenous
    REFINE_PRIMARY in REFINED_TOTAL.
  - HS 780199 (crude / unrefined) appears ONLY in CRUDE_supply, never in
    FEED_DOM. HS 780191 stays in FEED as refined.
  - HS 850790 (battery parts) is battery-committed and routes directly to
    MFG with eta_mfg only -- NOT subject to beta. Oxides (282410, 282490)
    stay in FEED and remain subject to beta.

v3 scripts remain untouched (fallback).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --- B1.1: segment config loader -----------------------------------------

def load_segments(path: str = 'india_model/segment_lifetimes.csv') -> pd.DataFrame:
    """Load segment lifetimes CSV and validate shares + lifetimes.

    Returns a DataFrame with columns: segment, stock_share, lifetime_years.
    Raises ValueError on invalid input.
    """
    df = pd.read_csv(path)
    required = {'segment', 'stock_share', 'lifetime_years'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'segment_lifetimes.csv missing columns: {sorted(missing)}')
    shares = df['stock_share'].to_numpy(dtype=float)
    lifetimes = df['lifetime_years'].to_numpy(dtype=float)
    if not np.isclose(shares.sum(), 1.0, atol=1e-6):
        raise ValueError(
            f'segment stock_share must sum to 1.0 (got {shares.sum():.6f})'
        )
    if not np.all(lifetimes > 0):
        raise ValueError(
            f'all lifetime_years must be > 0 (got {lifetimes.tolist()})'
        )
    return df


# --- B1.2: harmonic-mean effective lifetime ------------------------------

def tau_eff(segments: pd.DataFrame) -> float:
    """Stock-share-weighted harmonic mean of segment lifetimes.

        tau_eff = 1 / sum( w_s / tau_s )

    With the seed config (SLI 0.47/4y, e_rickshaw 0.11/3y, stationary 0.42/9y)
    this returns approximately 5.0 years.
    """
    shares = segments['stock_share'].to_numpy(dtype=float)
    lifetimes = segments['lifetime_years'].to_numpy(dtype=float)
    denom = float(np.sum(shares / lifetimes))
    return 1.0 / denom


# --- B1.3: stock growth rate ---------------------------------------------

def fit_growth_rate(stock_series, years) -> float:
    """Exponential growth rate g from log-linear regression of stock on year.

    g is the slope of ln(stock) vs year. g is invariant to a constant level
    multiplier (k_stock), so it can be computed once from the raw (unsmoothed)
    series and reused across all k_stock values in the profilers.
    """
    stock = np.asarray(stock_series, dtype=float)
    yrs = np.asarray(years, dtype=float)
    if np.any(stock <= 0):
        raise ValueError('fit_growth_rate requires strictly positive stock values')
    slope, _intercept = np.polyfit(yrs, np.log(stock), 1)
    return float(slope)


# --- B1.4: growth-corrected retirement -----------------------------------

def retire_rate(g: float, tau: float) -> float:
    """Growth-corrected retirement rate.

        retire_rate = g / (exp(g * tau) - 1)   for |g| > 1e-9
                    = 1 / tau                   otherwise (limit as g -> 0)

    Sanity checks:
        retire_rate(0.06, 5.0) ~= 0.1715
        retire_rate(0.0,  5.0) ==  0.20
    """
    if tau <= 0:
        raise ValueError(f'tau must be > 0 (got {tau})')
    if abs(g) <= 1e-9:
        return 1.0 / tau
    return g / (np.expm1(g * tau))


def retire_flow(eff_stock, g: float, tau: float) -> np.ndarray:
    """Per-year RETIRE(t) = eff_stock(t) * retire_rate(g, tau)."""
    rate = retire_rate(g, tau)
    return np.asarray(eff_stock, dtype=float) * rate


# --- B2: input loader ----------------------------------------------------

# FEED = refined-lead-equivalents that feed manufacturing AND are subject to
# the battery-share beta. 850790 (battery parts) moved OUT in Phase M -- parts
# are battery-committed and route directly to MFG with eta_mfg only.
# Oxides (282410, 282490) stay in FEED: litharge has genuine non-battery uses,
# so the beta discount is defensible there.
FEED_HS = [780110, 780191, 282410, 282490]
FEED_IMP_COLS = [f'imp_{hs}_t_pb' for hs in FEED_HS]
FEED_EXP_COLS = [f'exp_{hs}_t_pb' for hs in FEED_HS]

# PARTS = HS 850790 (battery parts / plates / grids). Battery-committed.
# Route fully to MFG (apply eta_mfg, NOT beta).
PARTS_HS = 850790

# CRUDE = post-smelt, pre-refine. HS 780199 (lead, "other unwrought") trade
# attaches here, NOT in FEED.
CRUDE_HS = 780199


def load_inputs(path: str = 'india_model/india_mass_balance_2018_2023.csv',
                smooth_window: int | None = None,
                trim_years=None) -> dict:
    """Load the bundled India mass-balance inputs and produce an `arr` dict.

    Three-stage chain: 780199 (crude) is kept SEPARATE from FEED. FEED only
    contains refined-equivalents subject to beta: {780110, 780191, 282410,
    282490}. 850790 (battery parts) is exported as a separate `imp_parts` /
    `exp_parts` pair so the forward chain can route it directly to MFG
    without the beta discount.

    Sets stock_pre to the raw one-year-before-first-window stock (matching
    the v3 smoothed convention, so dStock(t=first) anchors on the prior
    year's raw stock).

    smooth_window: if not None, apply centered rolling mean of that window
      to all numeric columns; trim to trim_years.
    trim_years: iterable of years to keep after smoothing. Required when
      smooth_window is set; ignored otherwise.
    """
    df = pd.read_csv(path).sort_values('year').reset_index(drop=True)
    if smooth_window is not None:
        if trim_years is None:
            raise ValueError('trim_years required when smooth_window is set')
        numeric = [c for c in df.columns if c != 'year']
        rolled = df[numeric].rolling(window=smooth_window, center=True,
                                     min_periods=smooth_window).mean()
        df_s = pd.concat([df['year'], rolled], axis=1)
        trim_years = list(trim_years)
        first_year = min(trim_years)
        pre_year = first_year - 1
        stock_pre = float(df.loc[df['year'] == pre_year, 'stock_total_t_pb'].iloc[0])
        df_s = df_s[df_s['year'].isin(trim_years)].reset_index(drop=True)
        df_used = df_s
    else:
        log_stk = np.log(df['stock_total_t_pb'].to_numpy(dtype=float))
        slope, intercept = np.polyfit(df['year'].to_numpy(dtype=float), log_stk, 1)
        first_year = int(df['year'].iloc[0])
        stock_pre = float(np.exp(slope * (first_year - 1) + intercept))
        df_used = df

    arr = {
        'year':      df_used['year'].to_numpy(dtype=int),
        'stock':     df_used['stock_total_t_pb'].to_numpy(dtype=float),
        'stock_pre': stock_pre,
        'mine_usgs': df_used['mine_pb_t_usgs'].to_numpy(dtype=float),
        'prim_usgs': df_used['primary_pb_t_usgs'].to_numpy(dtype=float),
        'sec_usgs':  df_used['secondary_pb_t_usgs'].to_numpy(dtype=float),
        'imp_ore':   df_used['imp_260700_t_pb'].to_numpy(dtype=float),
        'exp_ore':   df_used['exp_260700_t_pb'].to_numpy(dtype=float),
        'imp_feed':  df_used[FEED_IMP_COLS].sum(axis=1).to_numpy(dtype=float),
        'exp_feed':  df_used[FEED_EXP_COLS].sum(axis=1).to_numpy(dtype=float),
        'imp_parts': df_used[f'imp_{PARTS_HS}_t_pb'].to_numpy(dtype=float),
        'exp_parts': df_used[f'exp_{PARTS_HS}_t_pb'].to_numpy(dtype=float),
        'imp_crude': df_used[f'imp_{CRUDE_HS}_t_pb'].to_numpy(dtype=float),
        'exp_crude': df_used[f'exp_{CRUDE_HS}_t_pb'].to_numpy(dtype=float),
        'imp_batt':  (df_used['imp_850710_t_pb'] + df_used['imp_850720_t_pb']).to_numpy(dtype=float),
        'exp_batt':  (df_used['exp_850710_t_pb'] + df_used['exp_850720_t_pb']).to_numpy(dtype=float),
        'imp_used':  df_used['imp_854810_t_pb'].to_numpy(dtype=float),
        'exp_used':  df_used['exp_854810_t_pb'].to_numpy(dtype=float),
        'imp_scrap': df_used['imp_780200_t_pb'].to_numpy(dtype=float),
        'exp_scrap': df_used['exp_780200_t_pb'].to_numpy(dtype=float),
    }
    return arr


# --- B2: fixed inputs ----------------------------------------------------
# Total collection rate (Dalberg/USAID). The formal/informal split happens
# downstream at break/smelt, so GAMMA is the TOTAL collection rate.
GAMMA_TOTAL = 0.98
# Battery share of refined-lead demand (ILZSG India). 0.90 is available as a
# sensitivity case but is NOT fitted; this is a fixed model input.
BETA_DEFAULT = 0.75


# --- B2: three-stage forward chain ---------------------------------------

def forward_chain(arr: dict,
                  k_stock: float,
                  phi_break_f: float,
                  phi_smelt_f: float,
                  *,
                  g: float,
                  tau: float,
                  beta: float = BETA_DEFAULT,
                  gamma: float | None = None,
                  delta: float = 0.95,
                  eta_break_F: float = 0.95,
                  eta_break_I: float = 0.70,
                  eta_smelt_F: float = 0.97,
                  eta_smelt_I: float = 0.60,
                  eta_refine:  float = 0.99,
                  eta_mfg:     float = 0.98) -> dict:
    """Run the v4 three-stage forward mass-balance chain.

    Inputs
      arr           : dict from load_inputs(...)
      k_stock       : scalar multiplier on reported stock
      phi_break_f   : formal share of BREAKING in [0, 1]   (the most informal stage)
      phi_smelt_f   : formal share of SMELTING in [0, 1]   (less informal than break)
      g, tau        : growth rate (raw stock) and effective lifetime (harmonic mean)
      beta          : battery share of FEED -- FIXED model input, default 0.75

    Structural assumption (not enforced inside this function; caller
    asserts):  phi_break_f < phi_smelt_f < 1.

    Refine is fully formal (phi_refine = 1, eta_refine = 0.99). USGS
    primary refined enters as exogenous REFINE_PRIMARY. USGS secondary
    refined enters ONLY res_refine -- never the forward chain.

    HS 780199 (crude/unrefined) attaches between SMELT and REFINE as
    `imp_crude` / `exp_crude`. It is NOT in FEED.

    Returns a dict of per-year arrays + residuals (res_refine, res_install).
    """
    stock = arr['stock']
    eff_stock = stock * k_stock
    eff_stock_pre = float(arr['stock_pre']) * k_stock

    retire_r = retire_rate(g, tau)
    RETIRE = eff_stock * retire_r

    # collection (total gamma; split happens downstream)
    if gamma is None:
        gamma = GAMMA_TOTAL
    COLLECT = gamma * RETIRE
    USED_DOM = COLLECT + arr['imp_used'] - arr['exp_used']
    ud = np.maximum(USED_DOM, 0.0)

    # BREAK -- most informal stage; split = phi_break_f
    BREAK_f = ud * phi_break_f       * delta * eta_break_F
    BREAK_i = ud * (1.0 - phi_break_f) * delta * eta_break_I
    BREAK_total = BREAK_f + BREAK_i
    SCRAP_supply = BREAK_total + arr['imp_scrap'] - arr['exp_scrap']

    # SMELT -- crude/unrefined output; less informal than break; split = phi_smelt_f
    sp = np.maximum(SCRAP_supply, 0.0)
    SMELT_f = sp * phi_smelt_f       * eta_smelt_F
    SMELT_i = sp * (1.0 - phi_smelt_f) * eta_smelt_I
    SMELT_crude = SMELT_f + SMELT_i

    # HS 780199 (crude) attaches HERE -- between smelt and refine.
    CRUDE_supply = SMELT_crude + arr['imp_crude'] - arr['exp_crude']

    # REFINE -- fully formal (phi_refine = 1); the bottleneck.
    cs = np.maximum(CRUDE_supply, 0.0)
    REFINE_SEC = cs * eta_refine
    REFINE_PRIMARY = arr['prim_usgs']
    REFINED_TOTAL = REFINE_PRIMARY + REFINE_SEC

    # FEED -- refined-equivalents subject to beta (NOT 780199, NOT 850790).
    FEED_DOM = REFINED_TOTAL + arr['imp_feed'] - arr['exp_feed']
    # Battery parts (HS 850790) are battery-committed: route fully to MFG
    # with eta_mfg only -- NO beta discount.
    NET_PARTS = arr['imp_parts'] - arr['exp_parts']
    MFG = (np.maximum(FEED_DOM, 0.0) * beta * eta_mfg
           + np.maximum(NET_PARTS, 0.0) * eta_mfg)
    INSTALL_implied = MFG + arr['imp_batt'] - arr['exp_batt']

    prev_eff = np.concatenate([[eff_stock_pre], eff_stock[:-1]])
    dStock = eff_stock - prev_eff
    INSTALL_target = dStock + RETIRE

    res_refine  = (REFINE_SEC      - arr['sec_usgs'])     / arr['sec_usgs']
    res_install = (INSTALL_implied - INSTALL_target)      / INSTALL_target

    return {
        'k_stock': k_stock,
        'phi_break_f': phi_break_f, 'phi_smelt_f': phi_smelt_f,
        'beta': beta, 'gamma': gamma,
        'tau': tau, 'g': g, 'retire_rate': retire_r,
        'eff_stock': eff_stock, 'eff_stock_pre': eff_stock_pre,
        'RETIRE': RETIRE, 'COLLECT': COLLECT, 'USED_DOM': USED_DOM,
        'BREAK_f': BREAK_f, 'BREAK_i': BREAK_i, 'BREAK_total': BREAK_total,
        'SCRAP_supply': SCRAP_supply,
        'SMELT_f': SMELT_f, 'SMELT_i': SMELT_i, 'SMELT_crude': SMELT_crude,
        'CRUDE_supply': CRUDE_supply,
        'REFINE_SEC': REFINE_SEC, 'REFINE_PRIMARY': REFINE_PRIMARY,
        'REFINED_TOTAL': REFINED_TOTAL,
        'FEED_DOM': FEED_DOM, 'NET_PARTS': NET_PARTS, 'MFG': MFG,
        'INSTALL_implied': INSTALL_implied,
        'INSTALL_target': INSTALL_target, 'dStock': dStock,
        'res_refine': res_refine, 'res_install': res_install,
    }
