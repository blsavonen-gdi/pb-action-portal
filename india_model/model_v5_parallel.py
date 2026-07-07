"""India lead mass balance — model v5 (parallel formal/informal chain).

Successor to model_v4. Where v4 split formal/informal only at break and smelt
and treated refine + manufacture as fully formal, v5 runs both lanes through
ALL FOUR stages. There are four formal-share parameters now (not two):

    phi_break_f  <  phi_smelt_f  <  phi_refine_f  <  phi_mfg_f  <  1

Refining and manufacturing each have an informal lane with its own efficiency.
Crossover between lanes is implied (not solved): each stage's phi sets the
formal share of that stage's throughput, drawing from the combined output of
the upstream lanes. Because phi rises downstream, the formal lane pulls in
more than the upstream formal output produced -- that excess is informal
material "sold up" into formal, and we expose it as a derived quantity per
stage boundary.

Anchors (interpretation, not enforced inside this function):
  * USGS secondary is a ONE-SIDED FLOOR on formal refined output. Overshoot
    is expected and represents informal/unrecorded refined lead; only
    undershoot is a failure. (REFINE_SEC_F should be >= USGS_sec.)
  * Install side is a two-sided equality: INSTALL_implied vs INSTALL_target.

Trade attaches to the FORMAL lane only at each stage:
  * used-battery trade   (854810)
  * scrap trade          (780200)
  * crude trade          (780199)
  * FEED refined trade   (780110, 780191, 282410, 282490)
  * finished-battery trade (850710, 850720)

Primary lead (USGS primary) is also formal. The informal lane is purely
domestic.

Both lanes' MFG output flows into installation; INSTALL_implied = MFG_F +
MFG_I + imp_batt - exp_batt.

850790 (battery parts) is battery-committed: it routes directly to the formal
MFG output with eta_mfg_F, bypassing beta. Oxides (282410, 282490) remain in
FEED and remain subject to beta.

Public surface:
    REF_PHI, ETA_DEFAULTS, BETA_DEFAULT, GAMMA_TOTAL
    forward_parallel_chain(arr, k_stock, phi, *, g, tau, beta, gamma,
                           eta_break_F, eta_break_I, eta_smelt_F, eta_smelt_I,
                           eta_refine_F, eta_refine_I, eta_mfg_F, eta_mfg_I,
                           delta)
"""

from __future__ import annotations

import numpy as np

from india_model.model_v4 import retire_rate  # unchanged numerics


# ---- public defaults -------------------------------------------------------

# Battery share of refined-Pb demand (working India value).
BETA_DEFAULT = 0.86

# Total collection rate (Dalberg/USAID).
GAMMA_TOTAL = 0.98

# Reference phi-vector for the four stages. Ordered ascending; ceiling < 1.
# Starting values + per-stage floors come from USAID:
#   phi_mfg_f    >= 0.90  (start 0.95)
#   phi_refine_f >= 0.80  (start 0.90)
#   phi_smelt_f  >= 0.70  (start 0.80)
#   phi_break_f  start 0.70 (no floor below)
REF_PHI = {
    "phi_break_f":  0.70,
    "phi_smelt_f":  0.80,
    "phi_refine_f": 0.90,
    "phi_mfg_f":    0.95,
}

# Per-stage USAID-informed lower bounds on phi.
PHI_FLOORS = {
    "phi_break_f":  0.01,
    "phi_smelt_f":  0.70,
    "phi_refine_f": 0.80,
    "phi_mfg_f":    0.90,
}

# Efficiency defaults. Formal etas come from literature; informal etas are
# placeholders -- the dashboard exposes them as editable.
ETA_DEFAULTS = {
    "delta":        0.95,    # Pb remaining at end-of-life (shared)
    "eta_break_F":  0.95,
    "eta_break_I":  0.70,
    "eta_smelt_F":  0.97,
    "eta_smelt_I":  0.60,
    "eta_refine_F": 0.99,
    "eta_refine_I": 0.95,    # placeholder (was 0.88 in early runs)
    "eta_mfg_F":    0.98,
    "eta_mfg_I":    0.95,    # placeholder (was 0.85 in early runs)
}


# ---- the chain -------------------------------------------------------------

def forward_parallel_chain(
    arr: dict,
    k_stock: float,
    phi: dict,
    *,
    g: float,
    tau: float,
    beta: float = BETA_DEFAULT,
    gamma: float = GAMMA_TOTAL,
    delta: float = ETA_DEFAULTS["delta"],
    eta_break_F:  float = ETA_DEFAULTS["eta_break_F"],
    eta_break_I:  float = ETA_DEFAULTS["eta_break_I"],
    eta_smelt_F:  float = ETA_DEFAULTS["eta_smelt_F"],
    eta_smelt_I:  float = ETA_DEFAULTS["eta_smelt_I"],
    eta_refine_F: float = ETA_DEFAULTS["eta_refine_F"],
    eta_refine_I: float = ETA_DEFAULTS["eta_refine_I"],
    eta_mfg_F:    float = ETA_DEFAULTS["eta_mfg_F"],
    eta_mfg_I:    float = ETA_DEFAULTS["eta_mfg_I"],
) -> dict:
    """Forward parallel formal/informal chain at one (k, tau, phi, etas).

    arr is the dict produced by india_model.model_v4.load_inputs (or equivalent
    -- it must carry `year`, `stock`, `stock_pre`, `prim_usgs`, `sec_usgs`, and
    the trade arrays). phi is a dict with keys phi_break_f, phi_smelt_f,
    phi_refine_f, phi_mfg_f.

    The function does NOT enforce the ordering constraint; callers should.
    Implied informal->formal crossovers at each stage boundary are reported
    so the caller can flag negative crossovers (= ordering locally infeasible).

    Returns per-year arrays for every flow and stage, plus:
      - res_install_W (scalar window-sum residual, two-sided)
      - shortfall_refine_W (scalar window-sum, one-sided floor)
      - overshoot_refine_W (scalar window-sum, the implied unrecorded)
      - per-year residuals
      - crossovers per stage boundary (xover_smelt, xover_refine, xover_mfg)
    """
    phi_break_f  = float(phi["phi_break_f"])
    phi_smelt_f  = float(phi["phi_smelt_f"])
    phi_refine_f = float(phi["phi_refine_f"])
    phi_mfg_f    = float(phi["phi_mfg_f"])

    stock     = arr["stock"] * k_stock
    stock_pre = float(arr["stock_pre"]) * k_stock
    r = retire_rate(g, tau)

    # ---- shared retirement / collection ---------------------------------
    RETIRE  = stock * r
    COLLECT = gamma * RETIRE                              # single shared pool

    # ---- BREAK split (collected lead enters both lanes) -----------------
    # Used-battery trade (854810) is formal-only and attaches at the formal
    # break lane (imported used batteries flow through legal customs into
    # formal breakers; exports likewise leave the formal lane).
    in_break_F = COLLECT * phi_break_f + arr["imp_used"] - arr["exp_used"]
    in_break_I = COLLECT * (1.0 - phi_break_f)
    out_break_F = in_break_F * delta * eta_break_F
    out_break_I = in_break_I * delta * eta_break_I

    # ---- SCRAP supply (formal-only trade) -------------------------------
    scrap_dom_F  = out_break_F                            # formal portion
    scrap_dom_I  = out_break_I                            # informal portion
    scrap_total  = scrap_dom_F + scrap_dom_I + arr["imp_scrap"] - arr["exp_scrap"]
    # phi_smelt_f sets the formal share of TOTAL smelt throughput
    in_smelt_F = scrap_total * phi_smelt_f
    in_smelt_I = scrap_total * (1.0 - phi_smelt_f)
    out_smelt_F = in_smelt_F * eta_smelt_F
    out_smelt_I = in_smelt_I * eta_smelt_I

    # ---- CRUDE supply (formal-only 780199 trade) ------------------------
    crude_dom    = out_smelt_F + out_smelt_I
    crude_total  = crude_dom + arr["imp_crude"] - arr["exp_crude"]
    in_refine_F = crude_total * phi_refine_f
    in_refine_I = crude_total * (1.0 - phi_refine_f)
    REFINE_SEC_F = in_refine_F * eta_refine_F
    REFINE_SEC_I = in_refine_I * eta_refine_I

    # ---- REFINED pool (formal-only primary + FEED trade) ----------------
    refined_dom   = REFINE_SEC_F + REFINE_SEC_I
    refined_total = (refined_dom
                     + arr["prim_usgs"]
                     + arr["imp_feed"] - arr["exp_feed"])
    in_mfg_F = refined_total * phi_mfg_f
    in_mfg_I = refined_total * (1.0 - phi_mfg_f)
    NET_PARTS = arr["imp_parts"] - arr["exp_parts"]
    MFG_F = in_mfg_F * beta * eta_mfg_F + np.maximum(NET_PARTS, 0.0) * eta_mfg_F
    MFG_I = in_mfg_I * beta * eta_mfg_I
    MFG_total = MFG_F + MFG_I

    # ---- installation (both lanes' batteries flow to installs) ----------
    INSTALL_implied = MFG_total + arr["imp_batt"] - arr["exp_batt"]
    prev_stock = np.concatenate([[stock_pre], stock[:-1]])
    dStock = stock - prev_stock
    INSTALL_target = dStock + RETIRE

    # ---- implied informal->formal crossovers ----------------------------
    # Positive = informal material sold up into formal at this boundary.
    # Negative at smelt or refine = the ordering constraint is locally
    # infeasible there (formal lane would have to FLOW DOWN to informal,
    # which we don't model).
    xover_smelt  = in_smelt_F  - out_break_F
    xover_refine = in_refine_F - out_smelt_F
    # At mfg: formal mfg pulls from the formal refined POOL (which already
    # includes primary + feed inflows that route into formal). xover_mfg
    # may legitimately be negative if primary+feed is large; it does not
    # signal ordering infeasibility on its own.
    xover_mfg    = in_mfg_F    - REFINE_SEC_F

    # ---- residuals ------------------------------------------------------
    # Per-year (signed)
    res_install_per_year = (INSTALL_implied - INSTALL_target) / INSTALL_target
    # Refine: per-year shortfall (one-sided), overshoot reported as magnitude
    sec_usgs = arr["sec_usgs"]
    shortfall_refine_per_year = np.maximum(sec_usgs - REFINE_SEC_F, 0.0) / sec_usgs
    overshoot_refine_per_year = np.maximum(REFINE_SEC_F - sec_usgs, 0.0)

    # Window-sum scalars
    res_install_W = float(
        (INSTALL_implied.sum() - INSTALL_target.sum()) / INSTALL_target.sum()
    )
    refine_sum = float(REFINE_SEC_F.sum())
    usgs_sum   = float(sec_usgs.sum())
    shortfall_refine_W = max(0.0, (usgs_sum - refine_sum) / usgs_sum)
    overshoot_refine_W = max(0.0, (refine_sum - usgs_sum) / usgs_sum)
    overshoot_refine_W_tonnes = max(0.0, refine_sum - usgs_sum)

    return {
        # echo of inputs
        "k_stock": float(k_stock),
        "phi_break_f": phi_break_f, "phi_smelt_f": phi_smelt_f,
        "phi_refine_f": phi_refine_f, "phi_mfg_f": phi_mfg_f,
        "beta": float(beta), "gamma": float(gamma),
        "tau": float(tau), "g": float(g), "retire_rate": float(r),
        "eff_stock": stock, "eff_stock_pre": stock_pre,

        # shared
        "RETIRE": RETIRE, "COLLECT": COLLECT,

        # break
        "in_break_F": in_break_F, "in_break_I": in_break_I,
        "out_break_F": out_break_F, "out_break_I": out_break_I,
        "BREAK_total": out_break_F + out_break_I,

        # smelt
        "scrap_total": scrap_total,
        "in_smelt_F": in_smelt_F, "in_smelt_I": in_smelt_I,
        "out_smelt_F": out_smelt_F, "out_smelt_I": out_smelt_I,
        "SMELT_total": out_smelt_F + out_smelt_I,

        # refine
        "crude_total": crude_total,
        "in_refine_F": in_refine_F, "in_refine_I": in_refine_I,
        "REFINE_SEC_F": REFINE_SEC_F, "REFINE_SEC_I": REFINE_SEC_I,
        "REFINE_SEC_total": REFINE_SEC_F + REFINE_SEC_I,
        "REFINE_PRIMARY": arr["prim_usgs"],

        # refined pool / mfg
        "refined_total": refined_total,
        "in_mfg_F": in_mfg_F, "in_mfg_I": in_mfg_I,
        "MFG_F": MFG_F, "MFG_I": MFG_I, "MFG_total": MFG_total,
        "NET_PARTS": NET_PARTS,

        # install
        "INSTALL_implied": INSTALL_implied,
        "INSTALL_target":  INSTALL_target,
        "dStock":          dStock,

        # crossovers
        "xover_smelt":  xover_smelt,
        "xover_refine": xover_refine,
        "xover_mfg":    xover_mfg,

        # residuals
        "res_install_per_year":       res_install_per_year,
        "shortfall_refine_per_year":  shortfall_refine_per_year,
        "overshoot_refine_per_year":  overshoot_refine_per_year,
        "res_install_W":              res_install_W,
        "shortfall_refine_W":         shortfall_refine_W,
        "overshoot_refine_W":         overshoot_refine_W,
        "overshoot_refine_W_tonnes":  overshoot_refine_W_tonnes,
    }


# ---- ordering / crossover helpers -----------------------------------------

def phi_is_ordered(phi: dict, *, eps: float = 1e-4) -> bool:
    """True iff 0 < phi_break_f < phi_smelt_f < phi_refine_f < phi_mfg_f < 1."""
    p = (phi["phi_break_f"], phi["phi_smelt_f"],
         phi["phi_refine_f"], phi["phi_mfg_f"])
    return (p[0] > eps and p[0] + eps < p[1]
            and p[1] + eps < p[2] and p[2] + eps < p[3] and p[3] < 1.0 - eps)


def crossovers_nonneg(out: dict, atol: float = 1e-3) -> bool:
    """True iff implied formal-pull from previous formal output is non-negative
    at smelt and refine (xover_mfg is informational only; see chain notes)."""
    return bool(np.all(out["xover_smelt"]  >= -atol)
                and np.all(out["xover_refine"] >= -atol))
