"""
india_mass_balance.py — India lead-acid battery mass balance model.

All quantities in metric tonnes of Pb content.
τ (battery lifespan lag) is adjustable; default = 4 years.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

INDIA_DIR = Path(__file__).parent
TAU = 4  # default battery lifespan lag (years); overridable via prepare_model_inputs
INDIA_BACI_NAME = "India"

# HS code → model stage
HS_STAGE: dict[int, str] = {
    260700: "ore",
    780110: "refined",
    780191: "refined",
    780199: "smelted",   # other unwrought — enters secondary refining input
    850710: "use",
    850720: "use",
    854810: "collection",
    780200: "scrap",
    282410: "refined",
    282490: "refined",
}


def load_india_csvs() -> dict[str, pd.DataFrame]:
    """Load India-specific CSV files from india_model/ directory."""
    result = {}
    for key, filename in [
        ("install", "india_install_estimates.csv"),
        ("primary", "india_primary_refined.csv"),
        ("secondary", "india_secondary_refined.csv"),
        ("mining", "india_mining.csv"),
    ]:
        result[key] = pd.read_csv(INDIA_DIR / filename)
    return result


def _to_series(df: pd.DataFrame) -> dict[int, float]:
    """Extract {year: value} from a standard India CSV."""
    return {int(r["year"]): float(r["value"]) for _, r in df.iterrows()}


def build_net_trade(baci_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Filter BACI to India and compute net trade per stage per year.

    Returns (net_trade_df, warnings) where warnings lists HS codes absent
    from the data for any year.
    """
    mask = (baci_df["Importer"] == INDIA_BACI_NAME) | (baci_df["Exporter"] == INDIA_BACI_NAME)
    india = baci_df[mask].copy()
    india["stage"] = india["Product"].map(HS_STAGE)
    india = india.dropna(subset=["stage"])

    # Imports positive, exports negative
    india["net_sign"] = np.where(india["Importer"] == INDIA_BACI_NAME, 1.0, -1.0)
    india["net_lead"] = india["actual_lead"] * india["net_sign"]

    net = (
        india.groupby(["Year", "stage"])["net_lead"]
        .sum()
        .unstack(fill_value=0.0)
        .reset_index()
        .rename(columns={"Year": "year"})
    )

    stages = ["ore", "refined", "smelted", "use", "collection", "scrap"]
    warnings: list[str] = []
    for s in stages:
        if s not in net.columns:
            net[s] = 0.0
            warnings.append(f"No BACI data found for stage '{s}' — treated as 0.")

    return net[["year"] + stages], warnings


def prepare_model_inputs(
    india_dfs: dict[str, pd.DataFrame],
    net_trade_df: pd.DataFrame,
    tau: int = TAU,
) -> tuple[dict[int, dict], list[int], dict[int, int]]:
    """
    Build year-keyed input dict for the model.

    Args:
        tau: battery lifespan lag in years (default TAU = 4)

    Returns:
        data          — {year: input_dict} for all candidate years
        valid_years   — years where the tau-lag install value exists
        missing_lag   — {model_year: required_lag_year} for years that cannot run
    """
    install_s = _to_series(india_dfs["install"])
    primary_s = _to_series(india_dfs["primary"])
    secondary_s = _to_series(india_dfs["secondary"])

    nt = net_trade_df.set_index("year").to_dict(orient="index")

    candidate_years = sorted(set(primary_s) & set(secondary_s))

    valid_years: list[int] = []
    missing_lag: dict[int, int] = {}

    for y in candidate_years:
        lag_year = y - tau
        if lag_year in install_s:
            valid_years.append(y)
        else:
            missing_lag[y] = lag_year

    data: dict[int, dict] = {}
    for y in candidate_years:
        lag_year = y - tau
        nt_y = nt.get(y, {})
        data[y] = {
            "install_lag": install_s.get(lag_year),
            "install_obs": install_s.get(y, 0.0),
            "primary_refined": primary_s.get(y, 0.0),
            "secondary_refined_obs": secondary_s.get(y, 0.0),
            "net_trade_use": nt_y.get("use", 0.0),
            "net_trade_collection": nt_y.get("collection", 0.0),
            "net_trade_scrap": nt_y.get("scrap", 0.0),
            "net_trade_smelted": nt_y.get("smelted", 0.0),
            "net_trade_refined": nt_y.get("refined", 0.0),
            "lag_year": lag_year,
        }

    return data, valid_years, missing_lag


def run_year(t_data: dict, params: dict) -> dict:
    """
    Execute mass balance equations for a single year.

    params must contain: gamma_disposal, gamma_F, phi_B, phi_S,
    eta_break_F, eta_break_I, eta_smelt_F, eta_smelt_I, eta_refine,
    beta, lead_loss_rate
    """
    install_lag: float = t_data["install_lag"]
    primary_refined: float = t_data["primary_refined"]

    gamma_disposal: float = params["gamma_disposal"]
    gamma_total: float = 1.0 - gamma_disposal
    gamma_F: float = float(np.clip(params["gamma_F"], 0.0, gamma_total))
    gamma_I: float = gamma_total - gamma_F
    phi_B: float = float(np.clip(params["phi_B"], 0.0, 1.0))
    phi_S: float = float(np.clip(params["phi_S"], 0.0, 1.0))
    eta_break_F: float = params["eta_break_F"]
    eta_break_I: float = params["eta_break_I"]
    eta_smelt_F: float = params["eta_smelt_F"]
    eta_smelt_I: float = params["eta_smelt_I"]
    eta_refine: float = params["eta_refine"]
    beta: float = params["beta"]
    lead_loss_rate: float = params["lead_loss_rate"]

    # Step 1
    USE_eol = install_lag * (1.0 - lead_loss_rate)
    degradation_loss = install_lag * lead_loss_rate

    # Step 2
    COLL_F = gamma_F * USE_eol + t_data["net_trade_collection"]
    COLL_I = gamma_I * USE_eol
    DISPOSED = gamma_disposal * USE_eol

    # Step 3
    BREAK_F_in = COLL_F + phi_B * COLL_I
    BREAK_F_out = eta_break_F * BREAK_F_in
    BREAK_I_in = (1.0 - phi_B) * COLL_I
    BREAK_I_out = eta_break_I * BREAK_I_in
    BREAK_loss = (BREAK_F_in - BREAK_F_out) + (BREAK_I_in - BREAK_I_out)

    # Step 4
    SMELT_F_in = BREAK_F_out + phi_S * BREAK_I_out + t_data["net_trade_scrap"]
    SMELT_F_out = eta_smelt_F * SMELT_F_in
    SMELT_I_in = (1.0 - phi_S) * BREAK_I_out
    SMELT_I_out = eta_smelt_I * SMELT_I_in
    SMELT_loss = (SMELT_F_in - SMELT_F_out) + (SMELT_I_in - SMELT_I_out)

    # Step 5
    SEC_REFINE_in = SMELT_F_out + SMELT_I_out + t_data["net_trade_smelted"]
    SEC_REFINED = eta_refine * SEC_REFINE_in
    SEC_REFINE_loss = SEC_REFINE_in - SEC_REFINED

    # Step 6
    POOL = SEC_REFINED + primary_refined + t_data["net_trade_refined"]
    BATTERY_MFG = beta * POOL
    NON_BATTERY = (1.0 - beta) * POOL

    # Step 7
    implied_install = BATTERY_MFG + t_data["net_trade_use"]

    return {
        "USE_eol": USE_eol,
        "degradation_loss": degradation_loss,
        "COLL_F": COLL_F,
        "COLL_I": COLL_I,
        "DISPOSED": DISPOSED,
        "BREAK_F_in": BREAK_F_in,
        "BREAK_F_out": BREAK_F_out,
        "BREAK_I_in": BREAK_I_in,
        "BREAK_I_out": BREAK_I_out,
        "BREAK_loss": BREAK_loss,
        "SMELT_F_in": SMELT_F_in,
        "SMELT_F_out": SMELT_F_out,
        "SMELT_I_in": SMELT_I_in,
        "SMELT_I_out": SMELT_I_out,
        "SMELT_loss": SMELT_loss,
        "SEC_REFINE_in": SEC_REFINE_in,
        "SEC_REFINED": SEC_REFINED,
        "SEC_REFINE_loss": SEC_REFINE_loss,
        "POOL": POOL,
        "BATTERY_MFG": BATTERY_MFG,
        "NON_BATTERY": NON_BATTERY,
        "implied_install": implied_install,
        "gamma_F": gamma_F,
        "gamma_I": gamma_I,
        "gamma_total": gamma_total,
    }


def solve_model(
    data: dict[int, dict],
    valid_years: list[int],
    fixed_params: dict,
    w1: float = 1.0,
    w2: float = 1.0,
) -> dict:
    """
    Fit γ_F(t) per year plus φ_B and φ_S (year-invariant) by minimising:
        Σ_t [ w1 × (SEC_REFINED(t) − obs)² + w2 × (implied_install(t) − install(t))² ]

    Residuals are normalised internally to avoid SLSQP numerical issues on
    large absolute values. Uses scipy SLSQP with bounds and smoothness
    constraints on γ_F; falls back to L-BFGS-B (bounds only) if SLSQP
    reports infeasible constraints.
    """
    n = len(valid_years)
    if n == 0:
        return {
            "gamma_F": np.array([]),
            "phi_B": fixed_params.get("phi_B", 0.7),
            "phi_S": fixed_params.get("phi_S", 0.7),
            "success": False,
            "message": "No valid model years (install lag data missing).",
            "fun": float("nan"),
        }

    gamma_total = 1.0 - fixed_params["gamma_disposal"]

    # Normalisation scale: use mean observed values so residuals are ~O(1)
    _scale_sec = max(
        np.mean([data[y]["secondary_refined_obs"] for y in valid_years]), 1.0
    )
    _scale_inst = max(
        np.mean([data[y]["install_obs"] for y in valid_years]), 1.0
    )

    # φ_B and φ_S are now fixed by sliders; solver only fits γ_F(t)
    x0 = np.full(n, 0.6 * gamma_total)
    bounds = [(0.0, gamma_total)] * n

    # |γ_F(t) − γ_F(t−1)| ≤ 0.05 for consecutive years
    constraints = []
    for i in range(1, n):
        def _up(x, i=i):
            return 0.05 - (x[i] - x[i - 1])

        def _dn(x, i=i):
            return 0.05 - (x[i - 1] - x[i])

        constraints += [
            {"type": "ineq", "fun": _up},
            {"type": "ineq", "fun": _dn},
        ]

    def objective(x: np.ndarray) -> float:
        total = 0.0
        for i, y in enumerate(valid_years):
            p = {**fixed_params, "gamma_F": x[i]}
            r = run_year(data[y], p)
            r1 = (r["SEC_REFINED"] - data[y]["secondary_refined_obs"]) / _scale_sec
            r2 = (r["implied_install"] - data[y]["install_obs"]) / _scale_inst
            total += w1 * r1 ** 2 + w2 * r2 ** 2
        return total

    res = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 2000, "ftol": 1e-12},
    )

    # Fallback: if SLSQP reports infeasible constraints, retry without them
    if not res.success and "incompatible" in res.message.lower():
        res_fallback = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 2000},
        )
        if res_fallback.fun < res.fun:
            res = res_fallback
            res.message = (  # type: ignore[attr-defined]
                "SLSQP constraints infeasible; used L-BFGS-B (bounds only). "
                + res.message
            )

    return {
        "gamma_F": res.x[:n],
        "phi_B": fixed_params.get("phi_B", 0.7),
        "phi_S": fixed_params.get("phi_S", 0.7),
        "success": res.success,
        "message": res.message,
        "fun": float(objective(res.x)),
    }


def forward_model(
    data: dict[int, dict],
    valid_years: list[int],
    fixed_params: dict,
) -> pd.DataFrame:
    """
    Run forward model for all valid years using fixed_params.
    fixed_params must include gamma_F, phi_B, phi_S plus all η/β values.
    Returns DataFrame with all model quantities, residuals, and observations.
    """
    rows = []
    for y in valid_years:
        r = run_year(data[y], fixed_params)
        rows.append({
            "year": y,
            **r,
            "phi_B": fixed_params["phi_B"],
            "phi_S": fixed_params["phi_S"],
            # net trade by stage (positive = imports, negative = exports)
            "net_trade_use": data[y]["net_trade_use"],
            "net_trade_collection": data[y]["net_trade_collection"],
            "net_trade_scrap": data[y]["net_trade_scrap"],
            "net_trade_smelted": data[y]["net_trade_smelted"],
            "net_trade_refined": data[y]["net_trade_refined"],
            "secondary_refined_obs": data[y]["secondary_refined_obs"],
            "install_obs": data[y]["install_obs"],
            "residual_sec_refine": r["SEC_REFINED"] - data[y]["secondary_refined_obs"],
            "residual_install": r["implied_install"] - data[y]["install_obs"],
        })
    return pd.DataFrame(rows)
