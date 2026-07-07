"""Interactive calibration explorer for India v3 Scenario B.

Pure-function variant of calibrate_india_2018_2023.evaluate so efficiency
constants can be passed in (rather than mutating module globals). Exposes a
refit() that holds any subset of parameters pinned to user values while
SLSQP fits the rest under the same constraints as the headline calibration:
  - hard equality: INSTALL_implied(t) = INSTALL_target(t) for every year
  - inequality:    phi_smelt_f(t) - phi_break_f(t) >= 0.01
  - smelt-only objective + smoothness regularizer (lambda=10)

The calibration window is configurable via the `years` argument throughout,
so the Streamlit tab can drive it from the sidebar's active_years selection.
Inputs are built from baci_df + mining_df + Pb factors so sidebar adjusters
flow through naturally. The pre-period stock value (year before the window)
is sourced from the bundled india_mass_balance_2018_2023.csv stock column,
which is the only piece of data not derivable from the app-wide datasets.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

INDIA_DIR = Path(__file__).parent
INPUT_CSV = INDIA_DIR / "india_mass_balance_2018_2023.csv"
BEST_FIT_PARAMS_CSV = INDIA_DIR / "india_calibration_params_B_t5_smooth.csv"

INDIA_BACI_NAME = "India"
INDIA_MINING_NAME = "India"

# Default calibration window (matches the published Scenario B best fit).
DEFAULT_YEARS: tuple[int, ...] = (2019, 2020, 2021, 2022)

TV_PARAMS = ("gamma", "beta", "phi_break_f", "phi_smelt_f")

BOUNDS = {
    "tau":         (0.5, 30.0),
    "k_stock":     (0.05, 5.0),
    "gamma":       (0.96, 1.00),
    "beta":        (0.60, 0.90),
    "phi_break_f": (0.10, 0.95),
    "phi_smelt_f": (0.10, 0.95),
}

ETAS_DEFAULT = {
    "delta":         0.95,
    "eta_break_f":   0.95,
    "eta_break_i":   0.70,
    "eta_scrap_f":   0.97,
    "eta_scrap_i":   0.60,
    "eta_mfg":       0.98,
}

ETA_BOUNDS = {
    "delta":         (0.85, 1.00),
    "eta_break_f":   (0.80, 0.99),
    "eta_break_i":   (0.50, 0.95),
    "eta_scrap_f":   (0.85, 0.99),
    "eta_scrap_i":   (0.40, 0.90),
    "eta_mfg":       (0.90, 1.00),
}

CONSTRAINT_MARGIN = 0.01
SMOOTH_WINDOW = 3

# HS codes by model stage. 854911 is the HS22 equivalent of 854810 (waste
# batteries) — both map to the USED/collection stage.
HS_BY_STAGE: dict[str, tuple[int, ...]] = {
    "feed":  (780110, 780191, 780199, 850790, 282410, 282490),
    "batt":  (850710, 850720),
    "used":  (854810, 854911),
    "scrap": (780200,),
}

FEED_IMP_COLS = ['imp_780110_t_pb', 'imp_780191_t_pb', 'imp_780199_t_pb',
                 'imp_850790_t_pb', 'imp_282410_t_pb', 'imp_282490_t_pb']
FEED_EXP_COLS = ['exp_780110_t_pb', 'exp_780191_t_pb', 'exp_780199_t_pb',
                 'exp_850790_t_pb', 'exp_282410_t_pb', 'exp_282490_t_pb']


# --- Per-window context (replaces module-level YEARS/N_Y/PARAM_INDEX) ---

class CalibrationContext:
    """Carries year-dependent indexing and bound vectors for a given window."""
    __slots__ = ("years", "n_y", "n_params", "layout", "index")

    def __init__(self, years: Sequence[int]):
        self.years = tuple(int(y) for y in years)
        self.n_y = len(self.years)
        self.layout = ["tau", "k_stock"] + [
            f"{name}_{y}" for name in TV_PARAMS for y in self.years
        ]
        self.index = {name: i for i, name in enumerate(self.layout)}
        self.n_params = len(self.layout)

    def tv_slot(self, name: str, year: int) -> int:
        return self.index[f"{name}_{int(year)}"]

    def unpack(self, x: np.ndarray) -> dict:
        out = {"tau": float(x[0]), "k_stock": float(x[1])}
        off = 2
        for name in TV_PARAMS:
            out[name] = np.asarray(x[off:off + self.n_y], dtype=float)
            off += self.n_y
        return out


# --- Stock series helpers -----------------------------------------------

def load_india_stock_series() -> pd.Series:
    """Annual stock_total_t_pb for India, indexed by year. Source: bundled
    india_mass_balance_2018_2023.csv. Used both as the per-year stock and
    as the pre-period stock for dStock(min_year)."""
    df = pd.read_csv(INPUT_CSV)
    return df.set_index("year")["stock_total_t_pb"].astype(float)


# --- BACI aggregation ---------------------------------------------------

def _india_trade_by_stage_year(
    baci_df: pd.DataFrame,
    pb_factors: dict[int, float],
    stages: dict[str, tuple[int, ...]] = HS_BY_STAGE,
) -> pd.DataFrame:
    """Aggregate India's trade: returns DataFrame indexed by Year with one
    column per stage × direction (e.g., imp_feed, exp_feed, imp_batt, ...)
    in tonnes of Pb content. Quantity × pb_factors[hs]."""
    df = baci_df[
        (baci_df["Importer"] == INDIA_BACI_NAME) |
        (baci_df["Exporter"] == INDIA_BACI_NAME)
    ].copy()
    df["pb_t"] = df["Quantity"] * df["Product"].map(pb_factors).fillna(0.0)

    # All India-related rows are tagged with the same India side; the other side
    # is a partner country, so the direction is determined by which side India is on.
    cols: dict[str, pd.Series] = {}
    for stage, codes in stages.items():
        for direction, baci_col in (("imp", "Importer"), ("exp", "Exporter")):
            mask = df[baci_col].eq(INDIA_BACI_NAME) & df["Product"].isin(codes)
            agg = df.loc[mask].groupby("Year")["pb_t"].sum()
            cols[f"{direction}_{stage}"] = agg
    out = pd.DataFrame(cols).fillna(0.0)
    out.index.name = "year"
    return out.sort_index()


def _india_usgs_by_year(mining_df: pd.DataFrame) -> pd.DataFrame:
    """USGS primary and secondary refining for India, indexed by year."""
    sub = mining_df[mining_df["country"] == INDIA_MINING_NAME].copy()
    sub = sub.set_index("year")
    return sub[["refined_primary_usgs_t", "refined_secondary_usgs_t"]].astype(float)


# --- Build arr ----------------------------------------------------------

def build_arr_from_app(
    baci_df: pd.DataFrame,
    mining_df: pd.DataFrame,
    pb_factors: dict[int, float],
    active_years: Sequence[int],
    smooth_3yr: bool,
) -> tuple[dict, list[str]]:
    """Build the `arr` dict consumed by evaluate() from app-wide datasets.

    The smoothed variant gathers raw data for active_years ± 1 to support a
    3-year centered rolling mean, then trims to active_years. Stock comes
    from the bundled CSV; the pre-period stock is taken from min(active_years)-1.

    Returns (arr, warnings).
    """
    warnings: list[str] = []
    years_sorted = sorted(int(y) for y in active_years)

    # Gather wider raw window for smoothing
    pad = 1 if smooth_3yr else 0
    raw_y_lo = years_sorted[0] - pad
    raw_y_hi = years_sorted[-1] + pad

    trade = _india_trade_by_stage_year(baci_df, pb_factors)
    usgs  = _india_usgs_by_year(mining_df)
    stock = load_india_stock_series()

    raw_years = list(range(raw_y_lo, raw_y_hi + 1))
    trade = trade.reindex(raw_years).fillna(0.0)
    usgs  = usgs.reindex(raw_years)
    stock_raw = stock.reindex(raw_years)

    # Warn where USGS or stock are missing
    missing_usgs = [y for y in years_sorted if usgs.loc[y].isna().any()]
    if missing_usgs:
        warnings.append(
            f"USGS primary/secondary missing for {missing_usgs}; falling back to nearest available year."
        )
        usgs = usgs.ffill().bfill()
    missing_stock = [y for y in years_sorted if pd.isna(stock_raw.loc[y])]
    if missing_stock:
        warnings.append(
            f"Stock series missing for {missing_stock}; falling back to log-linear extrapolation from 2018–2023."
        )
        stock_known = stock.dropna()
        log_stk = np.log(stock_known.values)
        slope, intercept = np.polyfit(stock_known.index.to_numpy(dtype=float), log_stk, 1)
        for y in raw_years:
            if pd.isna(stock_raw.loc[y]):
                stock_raw.loc[y] = float(np.exp(slope * y + intercept))

    # Smoothing
    if smooth_3yr and len(raw_years) >= 3:
        wide = pd.concat([trade, usgs, stock_raw.rename("stock_total_t_pb")], axis=1)
        smoothed = wide.rolling(window=SMOOTH_WINDOW, center=True, min_periods=SMOOTH_WINDOW).mean()
        wide_trim = smoothed.loc[years_sorted].copy()
        # Some active years may not have enough neighbors to smooth; warn and fall back to raw.
        if wide_trim.isna().any().any():
            wide_trim = wide_trim.fillna(wide.loc[years_sorted])
            warnings.append("Some boundary years lacked neighbors for 3-yr smoothing; raw values used there.")
        trade_s = wide_trim[trade.columns]
        usgs_s  = wide_trim[usgs.columns]
        stock_s = wide_trim["stock_total_t_pb"]
    else:
        trade_s = trade.loc[years_sorted]
        usgs_s  = usgs.loc[years_sorted]
        stock_s = stock_raw.loc[years_sorted]

    # Pre-period stock = raw (unsmoothed) stock at (min_year - 1)
    pre_year = years_sorted[0] - 1
    if pre_year in stock.index:
        stock_pre = float(stock.loc[pre_year])
    else:
        stock_known = stock.dropna()
        log_stk = np.log(stock_known.values)
        slope, intercept = np.polyfit(stock_known.index.to_numpy(dtype=float), log_stk, 1)
        stock_pre = float(np.exp(slope * pre_year + intercept))
        warnings.append(f"Pre-period stock ({pre_year}) extrapolated.")

    arr = {
        "year":      np.array(years_sorted),
        "stock":     stock_s.to_numpy(dtype=float),
        "prim_usgs": usgs_s["refined_primary_usgs_t"].to_numpy(dtype=float),
        "sec_usgs":  usgs_s["refined_secondary_usgs_t"].to_numpy(dtype=float),
        "imp_feed":  trade_s["imp_feed"].to_numpy(dtype=float),
        "exp_feed":  trade_s["exp_feed"].to_numpy(dtype=float),
        "imp_batt":  trade_s["imp_batt"].to_numpy(dtype=float),
        "exp_batt":  trade_s["exp_batt"].to_numpy(dtype=float),
        "imp_used":  trade_s["imp_used"].to_numpy(dtype=float),
        "exp_used":  trade_s["exp_used"].to_numpy(dtype=float),
        "imp_scrap": trade_s["imp_scrap"].to_numpy(dtype=float),
        "exp_scrap": trade_s["exp_scrap"].to_numpy(dtype=float),
        "stock_pre": stock_pre,
    }
    return arr, warnings


# --- Pure-function evaluator --------------------------------------------

def evaluate(x: np.ndarray, arr: dict, etas: dict, ctx: CalibrationContext) -> dict:
    p = ctx.unpack(x)
    tau = p["tau"]
    k   = p["k_stock"]
    gamma = p["gamma"]
    beta  = p["beta"]
    pbf   = p["phi_break_f"]
    psf   = p["phi_smelt_f"]
    pbi   = 1.0 - pbf
    psi   = 1.0 - psf

    DELTA = etas["delta"]
    EB_F  = etas["eta_break_f"]
    EB_I  = etas["eta_break_i"]
    ES_F  = etas["eta_scrap_f"]
    ES_I  = etas["eta_scrap_i"]
    EMFG  = etas["eta_mfg"]

    eff_stock     = arr["stock"] * k
    eff_stock_pre = arr["stock_pre"] * k

    RETIRE   = eff_stock / tau
    COLLECT  = gamma * RETIRE
    USED_DOM = COLLECT + arr["imp_used"] - arr["exp_used"]
    ud = np.maximum(USED_DOM, 0.0)

    USED_DOM_f = ud * pbf
    USED_DOM_i = ud * pbi
    BREAK_f    = USED_DOM_f * DELTA * EB_F
    BREAK_i    = USED_DOM_i * DELTA * EB_I
    BREAK_total = BREAK_f + BREAK_i

    SCRAP_supply = BREAK_total + arr["imp_scrap"] - arr["exp_scrap"]
    sp = np.maximum(SCRAP_supply, 0.0)
    SCRAP_to_f = sp * psf
    SCRAP_to_i = sp * psi
    SMELT_SEC_f = SCRAP_to_f * ES_F
    SMELT_SEC_i = SCRAP_to_i * ES_I

    SMELT_PRIMARY = arr["prim_usgs"]
    SMELT_TOTAL   = SMELT_PRIMARY + SMELT_SEC_f + SMELT_SEC_i

    FEED_DOM        = SMELT_TOTAL + arr["imp_feed"] - arr["exp_feed"]
    MFG             = np.maximum(FEED_DOM, 0.0) * beta * EMFG
    INSTALL_implied = MFG + arr["imp_batt"] - arr["exp_batt"]

    prev_eff = np.concatenate([[eff_stock_pre], eff_stock[:-1]])
    dStock          = eff_stock - prev_eff
    INSTALL_target  = dStock + RETIRE

    # Guard against zero observations (would NaN the residual)
    sec_safe   = np.where(arr["sec_usgs"] > 0, arr["sec_usgs"], 1.0)
    inst_safe  = np.where(np.abs(INSTALL_target) > 1.0, INSTALL_target, 1.0)
    res_smelt = (SMELT_SEC_f + SMELT_SEC_i - arr["sec_usgs"]) / sec_safe
    res_inst  = (INSTALL_implied - INSTALL_target) / inst_safe

    return {
        "tau": tau, "k_stock": k,
        "gamma": gamma, "beta": beta,
        "phi_break_f": pbf, "phi_break_i": pbi,
        "phi_smelt_f": psf, "phi_smelt_i": psi,
        "eff_stock": eff_stock,
        "RETIRE": RETIRE, "COLLECT": COLLECT, "USED_DOM": USED_DOM,
        "BREAK_f": BREAK_f, "BREAK_i": BREAK_i, "BREAK_total": BREAK_total,
        "SCRAP_supply": SCRAP_supply, "SCRAP_to_f": SCRAP_to_f, "SCRAP_to_i": SCRAP_to_i,
        "SMELT_SEC_f": SMELT_SEC_f, "SMELT_SEC_i": SMELT_SEC_i,
        "SMELT_SEC_total": SMELT_SEC_f + SMELT_SEC_i,
        "SMELT_TOTAL": SMELT_TOTAL, "SMELT_PRIMARY": SMELT_PRIMARY,
        "FEED_DOM": FEED_DOM, "MFG": MFG,
        "INSTALL_implied": INSTALL_implied, "INSTALL_target": INSTALL_target,
        "dStock": dStock,
        "res_smelt_pct": res_smelt,
        "res_install_pct": res_inst,
    }


# --- Objective & constraints --------------------------------------------

def smoothness(x: np.ndarray, lam: float, ctx: CalibrationContext) -> float:
    if ctx.n_y <= 1:
        return 0.0
    p = ctx.unpack(x)
    total = 0.0
    for name in TV_PARAMS:
        lo, hi = BOUNDS[name]
        rng = hi - lo
        norm = (p[name] - lo) / rng
        d = np.diff(norm)
        total += float(np.sum(d * d))
    return lam * total


ANCHORS = ("forward", "reversed", "balanced")

W_SMELT_BAL   = 1.0    # weight on smelt residual in balanced objective
W_INSTALL_BAL = 1.0    # weight on install residual in balanced objective


def _objective_forward(x, arr, etas, lam, ctx):
    out = evaluate(x, arr, etas, ctx)
    smelt_sse = float(np.sum(out["res_smelt_pct"] ** 2))
    return smelt_sse + smoothness(x, lam, ctx)


def _objective_reversed(x, arr, etas, lam, ctx):
    out = evaluate(x, arr, etas, ctx)
    inst_sse = float(np.sum(out["res_install_pct"] ** 2))
    return inst_sse + smoothness(x, lam, ctx)


def _objective_balanced(x, arr, etas, lam, ctx):
    out = evaluate(x, arr, etas, ctx)
    smelt_sse = float(np.sum(out["res_smelt_pct"] ** 2))
    inst_sse  = float(np.sum(out["res_install_pct"] ** 2))
    return W_SMELT_BAL * smelt_sse + W_INSTALL_BAL * inst_sse + smoothness(x, lam, ctx)


# Back-compat alias (used by tests and forward_only); kept pointing at the
# forward objective so external callers continue to work.
def objective(x, arr, etas, lam, ctx):
    return _objective_forward(x, arr, etas, lam, ctx)


def _phi_inequality_cons(ctx: CalibrationContext) -> list:
    """phi_smelt_f(t) - phi_break_f(t) >= CONSTRAINT_MARGIN for every year.
    Applied in all three anchor variants."""
    cons = []
    for y in ctx.years:
        i_pbf = ctx.tv_slot("phi_break_f", y)
        i_psf = ctx.tv_slot("phi_smelt_f", y)
        cons.append({
            "type": "ineq",
            "fun":  (lambda x, i_psf=i_psf, i_pbf=i_pbf:
                     x[i_psf] - x[i_pbf] - CONSTRAINT_MARGIN),
        })
    return cons


def _install_equality_cons(arr: dict, etas: dict, ctx: CalibrationContext) -> list:
    """INSTALL_implied(t) = INSTALL_target(t) per year (forward anchor)."""
    cons = []
    def fn(t):
        def f(x, t=t):
            out = evaluate(x, arr, etas, ctx)
            return (out["INSTALL_implied"][t] - out["INSTALL_target"][t]) * 1e-5
        return f
    for t in range(ctx.n_y):
        cons.append({"type": "eq", "fun": fn(t)})
    return cons


def _smelt_equality_cons(arr: dict, etas: dict, ctx: CalibrationContext) -> list:
    """SMELT_SEC_f(t) + SMELT_SEC_i(t) = sec_usgs(t) per year (reversed anchor)."""
    cons = []
    def fn(t):
        def f(x, t=t):
            out = evaluate(x, arr, etas, ctx)
            return (out["SMELT_SEC_f"][t] + out["SMELT_SEC_i"][t]
                    - arr["sec_usgs"][t]) * 1e-5
        return f
    for t in range(ctx.n_y):
        cons.append({"type": "eq", "fun": fn(t)})
    return cons


def _build_constraints(arr: dict, etas: dict, ctx: CalibrationContext,
                       anchor: str = "forward") -> list:
    """Constraint list for the chosen anchor variant.
      forward:  phi inequality + install equality
      reversed: phi inequality + smelt   equality
      balanced: phi inequality only (no equality)
    """
    if anchor not in ANCHORS:
        raise ValueError(f"Unknown anchor: {anchor!r}; expected one of {ANCHORS}")
    cons = _phi_inequality_cons(ctx)
    if anchor == "forward":
        cons.extend(_install_equality_cons(arr, etas, ctx))
    elif anchor == "reversed":
        cons.extend(_smelt_equality_cons(arr, etas, ctx))
    # balanced: phi only
    return cons


def _build_bounds_vec(pins: dict[str, float], ctx: CalibrationContext) -> list[tuple[float, float]]:
    bnds: list[tuple[float, float]] = []
    eps_abs = 1e-4
    for name in ctx.layout:
        base = name if name in ("tau", "k_stock") else name.rsplit("_", 1)[0]
        lo, hi = BOUNDS[base]
        if name in pins:
            v = float(np.clip(float(pins[name]), lo, hi))
            bnds.append((max(lo, v - eps_abs), min(hi, v + eps_abs)))
        else:
            bnds.append((lo, hi))
    return bnds


def _clip_x_to_bounds(x: np.ndarray, bnds: list[tuple[float, float]]) -> np.ndarray:
    x = x.astype(float).copy()
    for j, (lo, hi) in enumerate(bnds):
        if x[j] < lo:
            x[j] = lo
        if x[j] > hi:
            x[j] = hi
    return x


def _default_x0(ctx: CalibrationContext) -> np.ndarray:
    """Sensible starting point for the optimizer: tau=5, k=1, midpoints of bounds."""
    x = np.zeros(ctx.n_params)
    x[0] = 5.0
    x[1] = 1.0
    off = 2
    starts = {"gamma": 0.98, "beta": 0.75, "phi_break_f": 0.55, "phi_smelt_f": 0.65}
    for name in TV_PARAMS:
        x[off:off + ctx.n_y] = starts[name]
        off += ctx.n_y
    return x


# --- Re-fit -------------------------------------------------------------

def refit(
    arr: dict,
    etas: dict,
    ctx: CalibrationContext,
    pins: dict[str, float] | None = None,
    x_warmstart: np.ndarray | None = None,
    lam: float = 10.0,
    n_restarts: int = 4,
    seed: int = 0,
    feasible_tol_t: float = 5000.0,
    anchor: str = "forward",
) -> dict:
    """SLSQP fit with the chosen anchor variant.

    anchor:
      "forward"  -- hard install equality, minimize smelt residual + smoothness
      "reversed" -- hard smelt equality,   minimize install residual + smoothness
      "balanced" -- no hard equality, soft smelt+install residuals + smoothness
    """
    if anchor not in ANCHORS:
        raise ValueError(f"Unknown anchor: {anchor!r}; expected one of {ANCHORS}")
    pins = dict(pins or {})
    bnds = _build_bounds_vec(pins, ctx)
    cons = _build_constraints(arr, etas, ctx, anchor=anchor)

    if x_warmstart is None or len(x_warmstart) != ctx.n_params:
        x_warmstart = _default_x0(ctx)
    x_warmstart = _clip_x_to_bounds(np.asarray(x_warmstart, dtype=float), bnds)
    for name, val in pins.items():
        i = ctx.index[name]
        x_warmstart[i] = float(np.clip(val, bnds[i][0], bnds[i][1]))

    rng = np.random.default_rng(seed)
    starts = [x_warmstart]
    for _ in range(max(0, n_restarts - 1)):
        x0 = x_warmstart.copy()
        for j, (lo, hi) in enumerate(bnds):
            if hi - lo > 1e-3:
                x0[j] = rng.uniform(lo, hi)
        for y in ctx.years:
            i_pbf = ctx.tv_slot("phi_break_f", y)
            i_psf = ctx.tv_slot("phi_smelt_f", y)
            lo_psf, hi_psf = bnds[i_psf]
            if hi_psf - lo_psf > 1e-3 and x0[i_psf] < x0[i_pbf] + CONSTRAINT_MARGIN:
                x0[i_psf] = min(hi_psf, x0[i_pbf] + CONSTRAINT_MARGIN + 0.02)
        starts.append(_clip_x_to_bounds(x0, bnds))

    obj_fn = {
        "forward":  _objective_forward,
        "reversed": _objective_reversed,
        "balanced": _objective_balanced,
    }[anchor]

    # For balanced (no equality), "feasibility" doesn't apply -- treat all
    # successful solutions as feasible and choose by lowest objective.
    has_equality = anchor in ("forward", "reversed")

    best = None
    for x0 in starts:
        try:
            res = minimize(
                obj_fn, x0, args=(arr, etas, lam, ctx),
                method="SLSQP", bounds=bnds, constraints=cons,
                options={"maxiter": 600, "ftol": 1e-10},
            )
        except Exception:
            continue
        if not np.isfinite(res.fun):
            continue
        out = evaluate(res.x, arr, etas, ctx)
        inst_gap = float(np.max(np.abs(out["INSTALL_implied"] - out["INSTALL_target"])))
        smelt_gap = float(np.max(np.abs(out["SMELT_SEC_f"] + out["SMELT_SEC_i"] - arr["sec_usgs"])))

        if anchor == "forward":
            eq_viol = inst_gap
        elif anchor == "reversed":
            eq_viol = smelt_gap
        else:
            eq_viol = 0.0  # not enforced

        feasible = (not has_equality) or (eq_viol <= feasible_tol_t)
        attach = (res, out, eq_viol, feasible, inst_gap, smelt_gap)
        if best is None:
            best = attach
            continue
        _, _, prev_viol, prev_feas, _, _ = best
        if has_equality:
            if feasible and not prev_feas:
                best = attach
            elif feasible and prev_feas and res.fun < best[0].fun:
                best = attach
            elif (not feasible) and (not prev_feas) and eq_viol < prev_viol:
                best = attach
        else:
            if res.fun < best[0].fun:
                best = attach

    if best is None:
        return {"success": False, "message": "Optimizer failed across all restarts."}

    res, out, eq_viol, feasible, inst_gap, smelt_gap = best
    mae_smelt = float(np.mean(np.abs(out["res_smelt_pct"]))) * 100.0
    mae_inst  = float(np.mean(np.abs(out["res_install_pct"]))) * 100.0
    smelt_sse = float(np.sum(out["res_smelt_pct"] ** 2))
    inst_sse  = float(np.sum(out["res_install_pct"] ** 2))
    smooth_v  = smoothness(res.x, lam, ctx)

    if anchor == "forward":
        j_total = smelt_sse + smooth_v
    elif anchor == "reversed":
        j_total = inst_sse + smooth_v
    else:
        j_total = W_SMELT_BAL * smelt_sse + W_INSTALL_BAL * inst_sse + smooth_v

    return {
        "success": True,
        "x": res.x,
        "out": out,
        "anchor": anchor,
        "mae_smelt_pct": mae_smelt,
        "mae_install_pct": mae_inst,
        "smelt_sse": smelt_sse,
        "install_sse": inst_sse,
        "smoothness": smooth_v,
        "J_total": j_total,
        # Constraint-slack metrics (interpretation depends on anchor).
        "max_eq_violation_t_pb": eq_viol,
        "max_install_gap_t_pb":  inst_gap,
        "max_smelt_gap_t_pb":    smelt_gap,
        "feasible": feasible,
        "message": str(res.message),
        "etas": dict(etas),
    }


def forward_only(x: np.ndarray, arr: dict, etas: dict, ctx: CalibrationContext,
                 lam: float = 10.0, anchor: str = "forward") -> dict:
    out = evaluate(x, arr, etas, ctx)
    mae_smelt = float(np.mean(np.abs(out["res_smelt_pct"]))) * 100.0
    mae_inst  = float(np.mean(np.abs(out["res_install_pct"]))) * 100.0
    smelt_sse = float(np.sum(out["res_smelt_pct"] ** 2))
    inst_sse  = float(np.sum(out["res_install_pct"] ** 2))
    smooth_v  = smoothness(x, lam, ctx)
    inst_gap  = float(np.max(np.abs(out["INSTALL_implied"] - out["INSTALL_target"])))
    smelt_gap = float(np.max(np.abs(out["SMELT_SEC_f"] + out["SMELT_SEC_i"] - arr["sec_usgs"])))

    if anchor == "forward":
        eq_viol = inst_gap
        j_total = smelt_sse + smooth_v
    elif anchor == "reversed":
        eq_viol = smelt_gap
        j_total = inst_sse + smooth_v
    else:
        eq_viol = 0.0
        j_total = W_SMELT_BAL * smelt_sse + W_INSTALL_BAL * inst_sse + smooth_v

    return {
        "success": True,
        "x": x.copy(),
        "out": out,
        "anchor": anchor,
        "mae_smelt_pct": mae_smelt,
        "mae_install_pct": mae_inst,
        "smelt_sse": smelt_sse,
        "install_sse": inst_sse,
        "smoothness": smooth_v,
        "J_total": j_total,
        "max_eq_violation_t_pb": eq_viol,
        "max_install_gap_t_pb":  inst_gap,
        "max_smelt_gap_t_pb":    smelt_gap,
        "feasible": True,
        "message": "forward-only (no re-fit)",
        "etas": dict(etas),
    }


# --- Legacy helpers (still used to validate against the published best fit) -

def load_smoothed_arr(path: Path | str = INPUT_CSV) -> dict:
    """3-year centered rolling mean over the full 2018-2023 input, trimmed
    to 2019-2022 (the published Scenario B window). Kept for tests and the
    bundled-CSV path; the live tab uses build_arr_from_app."""
    df = pd.read_csv(path).sort_values("year").reset_index(drop=True)
    numeric_cols = [c for c in df.columns if c != "year"]
    smoothed = (
        df[numeric_cols]
        .rolling(window=SMOOTH_WINDOW, center=True, min_periods=SMOOTH_WINDOW)
        .mean()
    )
    df_s = pd.concat([df["year"], smoothed], axis=1)
    df_s = df_s[df_s["year"].isin(DEFAULT_YEARS)].reset_index(drop=True)
    raw_stock_2018 = float(df.loc[df["year"] == 2018, "stock_total_t_pb"].iloc[0])
    return {
        "year":      df_s["year"].to_numpy(),
        "stock":     df_s["stock_total_t_pb"].to_numpy(dtype=float),
        "prim_usgs": df_s["primary_pb_t_usgs"].to_numpy(dtype=float),
        "sec_usgs":  df_s["secondary_pb_t_usgs"].to_numpy(dtype=float),
        "imp_feed":  df_s[FEED_IMP_COLS].sum(axis=1).to_numpy(dtype=float),
        "exp_feed":  df_s[FEED_EXP_COLS].sum(axis=1).to_numpy(dtype=float),
        "imp_batt":  (df_s["imp_850710_t_pb"] + df_s["imp_850720_t_pb"]).to_numpy(dtype=float),
        "exp_batt":  (df_s["exp_850710_t_pb"] + df_s["exp_850720_t_pb"]).to_numpy(dtype=float),
        "imp_used":  df_s["imp_854810_t_pb"].to_numpy(dtype=float),
        "exp_used":  df_s["exp_854810_t_pb"].to_numpy(dtype=float),
        "imp_scrap": df_s["imp_780200_t_pb"].to_numpy(dtype=float),
        "exp_scrap": df_s["exp_780200_t_pb"].to_numpy(dtype=float),
        "stock_pre": raw_stock_2018,
    }


def load_best_fit_x(ctx: CalibrationContext | None = None) -> np.ndarray:
    """Parse india_calibration_params_B_t5_smooth.csv into the flat x vector
    for the published window (DEFAULT_YEARS). If a different ctx is given,
    returns a default x0 (best fit must be recomputed for non-default windows)."""
    if ctx is None:
        ctx = CalibrationContext(DEFAULT_YEARS)
    if tuple(ctx.years) != tuple(DEFAULT_YEARS):
        return _default_x0(ctx)
    p = pd.read_csv(BEST_FIT_PARAMS_CSV)
    by_param: dict = {}
    for _, r in p.iterrows():
        name = r["param"]
        yr = r["year"]
        if pd.isna(yr) or yr == "":
            by_param[name] = float(r["value"])
        else:
            by_param.setdefault(name, {})[int(float(yr))] = float(r["value"])
    x = np.zeros(ctx.n_params)
    x[0] = by_param["tau"]
    x[1] = by_param["k_stock"]
    off = 2
    for name in TV_PARAMS:
        for t, yr in enumerate(ctx.years):
            x[off + t] = by_param[name][int(yr)]
        off += ctx.n_y
    return x


def load_best_fit_etas() -> dict:
    p = pd.read_csv(BEST_FIT_PARAMS_CSV)
    out = dict(ETAS_DEFAULT)
    mapping = {
        "delta": "delta",
        "eta_break_formal": "eta_break_f",
        "eta_break_informal": "eta_break_i",
        "eta_scrap_formal": "eta_scrap_f",
        "eta_scrap_informal": "eta_scrap_i",
        "eta_mfg": "eta_mfg",
    }
    for _, r in p.iterrows():
        if r["param"] in mapping:
            out[mapping[r["param"]]] = float(r["value"])
    return out
