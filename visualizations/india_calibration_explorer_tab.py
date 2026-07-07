"""Streamlit tab: India v3 Mass Balance explorer (Scenario B convention, tau=5).

Sidebar adjusters that flow through:
  * BACI dataset      → baci_df argument
  * Pb content factors → applied during BACI aggregation
  * Year + Time Period → calibration window (active_years), 3-yr smoothing toggle
  * Mining/refining source: noted in UI only — India only has primary/secondary
    refining splits from USGS, so this tab uses USGS regardless.

UI contract (unchanged from the previous version):
  * tau, k_stock, gamma, beta, and the six eta/delta efficiencies are user-adjustable.
    phi_break_f and phi_smelt_f are always optimizer-driven.
  * Year selector controls which year the per-year gamma/beta sliders edit.
  * Re-fit button runs SLSQP holding pinned slots tight; other slots re-optimize.
  * Reset button restores the published best fit (Scenario B: tau=5 pinned).
  * Lock k/tau ratio toggle keeps k/tau constant when the user moves tau.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from india_model.india_calibration_explorer import (
    ANCHORS,
    BOUNDS,
    ETA_BOUNDS,
    ETAS_DEFAULT,
    CalibrationContext,
    build_arr_from_app,
    forward_only,
    load_best_fit_etas,
    refit,
)


SESSION_KEY = "india_explorer_state"

ANCHOR_LABELS = {
    "forward":  "Forward (anchor install; minimize smelt)",
    "reversed": "Reversed (anchor USGS smelt; minimize install)",
    "balanced": "Balanced (no hard anchor; soft on both)",
}

ANCHOR_HELP = {
    "forward":
        "Enforces INSTALL_implied = INSTALL_target every year as a hard "
        "constraint; the smelt residual is what gets minimized. Tends to "
        "rail tau or k_stock to satisfy install when the two anchors "
        "disagree.",
    "reversed":
        "Enforces SMELT_SEC_total = USGS_secondary every year as a hard "
        "constraint; the install residual is what gets minimized. Often "
        "yields more interpretable tau/k_stock values but exposes a "
        "structural install gap.",
    "balanced":
        "No hard equality. Both smelt and install residuals enter the "
        "objective as soft penalties (equal weight). Usually finds a "
        "Pareto-optimal point with both residuals under 5%.",
}


def _fingerprint(
    dataset_key: str,
    pb_factors: dict,
    active_years: list[int],
    time_period: str,
    mining_source: str,
) -> tuple:
    return (
        dataset_key,
        tuple(sorted((int(k), round(float(v), 6)) for k, v in pb_factors.items())),
        tuple(sorted(int(y) for y in active_years)),
        time_period,
        mining_source,
    )


def _compute_best_fit(arr: dict, etas: dict, ctx: CalibrationContext,
                      anchor: str = "forward") -> dict:
    """Best fit for the given window with tau pinned at 5 (Scenario B convention)."""
    return refit(
        arr, etas, ctx,
        pins={"tau": 5.0},
        n_restarts=8,
        seed=42,
        anchor=anchor,
    )


def _compute_all_best_fits(arr: dict, etas: dict, ctx: CalibrationContext) -> dict[str, dict]:
    """Compute the best fit for each anchor variant, holding tau=5 pinned."""
    out = {}
    for a in ANCHORS:
        out[a] = _compute_best_fit(arr, etas, ctx, anchor=a)
    return out


def _init_state_for(
    arr: dict,
    etas_best: dict,
    best_by_anchor: dict[str, dict],
    anchor: str,
    ctx: CalibrationContext,
    fingerprint: tuple,
) -> dict:
    best = best_by_anchor[anchor]
    x_best = best["x"]
    return {
        "fingerprint": fingerprint,
        "arr": arr,
        "ctx": ctx,
        "etas_best": dict(etas_best),
        "anchor": anchor,
        "best_by_anchor": best_by_anchor,
        "best": best,
        "current": best,
        "ratio_lock": False,
        "k_over_tau_locked": float(x_best[ctx.index["k_stock"]] / x_best[ctx.index["tau"]]),
        "slider_tau":   float(x_best[ctx.index["tau"]]),
        "slider_k":     float(x_best[ctx.index["k_stock"]]),
        "slider_gamma": {int(y): float(x_best[ctx.tv_slot("gamma", int(y))]) for y in ctx.years},
        "slider_beta":  {int(y): float(x_best[ctx.tv_slot("beta",  int(y))]) for y in ctx.years},
        "slider_etas":  dict(etas_best),
    }


def _switch_anchor(state: dict, new_anchor: str) -> None:
    """Switch the active anchor: best -> precomputed fit for that anchor,
    current -> best, and reset sliders to the new best."""
    if new_anchor not in state["best_by_anchor"]:
        return
    state["anchor"] = new_anchor
    state["best"]   = state["best_by_anchor"][new_anchor]
    _reset_to_best(state)


def _reset_to_best(state: dict) -> None:
    ctx: CalibrationContext = state["ctx"]
    x = state["best"]["x"]
    state["current"] = state["best"]
    state["slider_tau"] = float(x[ctx.index["tau"]])
    state["slider_k"]   = float(x[ctx.index["k_stock"]])
    state["slider_gamma"] = {int(y): float(x[ctx.tv_slot("gamma", int(y))]) for y in ctx.years}
    state["slider_beta"]  = {int(y): float(x[ctx.tv_slot("beta",  int(y))]) for y in ctx.years}
    state["slider_etas"]  = dict(state["etas_best"])
    state["k_over_tau_locked"] = float(x[ctx.index["k_stock"]] / x[ctx.index["tau"]])
    state["ratio_lock"] = False
    for key in list(st.session_state.keys()):
        if isinstance(key, str) and key.startswith("ice_"):
            del st.session_state[key]


def _do_refit(state: dict, sel_year: int) -> None:
    ctx: CalibrationContext = state["ctx"]
    pins = {
        "tau":     float(state["slider_tau"]),
        "k_stock": float(state["slider_k"]),
        f"gamma_{int(sel_year)}": float(state["slider_gamma"][int(sel_year)]),
        f"beta_{int(sel_year)}":  float(state["slider_beta"][int(sel_year)]),
    }
    etas = dict(state["slider_etas"])
    result = refit(
        state["arr"], etas, ctx,
        pins=pins,
        x_warmstart=state["current"].get("x"),
        n_restarts=4,
        anchor=state.get("anchor", "forward"),
    )
    if result.get("success"):
        state["current"] = result
        x_fit = result["x"]
        for y in ctx.years:
            yi = int(y)
            if yi == int(sel_year):
                continue
            state["slider_gamma"][yi] = float(x_fit[ctx.tv_slot("gamma", yi)])
            state["slider_beta"][yi]  = float(x_fit[ctx.tv_slot("beta",  yi)])
        state["slider_tau"] = float(x_fit[ctx.index["tau"]])
        state["slider_k"]   = float(x_fit[ctx.index["k_stock"]])
        # Drop widget session keys so the next render re-initializes sliders
        # from `value=state["slider_*"]`. Streamlit forbids __setitem__ on
        # widget keys after instantiation in the same run, but __delitem__
        # is allowed and is the only way to force a slider to display a
        # programmatically updated value on the next rerun.
        widget_keys = {"ice_tau", "ice_k"}
        for y in ctx.years:
            widget_keys.add(f"ice_gamma_{int(y)}")
            widget_keys.add(f"ice_beta_{int(y)}")
        for key in widget_keys:
            if key in st.session_state:
                del st.session_state[key]


def _residual_chart(current: dict, best: dict, residual_key: str, title: str, ylabel: str, years: tuple) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(years),
        y=(best["out"][residual_key] * 100).tolist(),
        name="Best fit",
        mode="lines+markers",
        line=dict(color="#666", dash="dash"),
        marker=dict(size=7),
    ))
    fig.add_trace(go.Scatter(
        x=list(years),
        y=(current["out"][residual_key] * 100).tolist(),
        name="Current",
        mode="lines+markers",
        line=dict(color="#1f77b4", width=3),
        marker=dict(size=9),
    ))
    fig.add_hline(y=0, line_color="#999", line_width=1)
    fig.update_layout(
        title=title,
        yaxis_title=ylabel,
        xaxis_title="Year",
        height=340,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _flow_chart(current: dict, best: dict, years: tuple, arr: dict) -> go.Figure:
    keys = [
        ("SMELT_SEC_total", "Secondary smelting (modeled)"),
        ("MFG", "Manufacturing"),
        ("INSTALL_implied", "Implied install"),
        ("BREAK_total", "Breaking (total)"),
    ]
    fig = go.Figure()
    for k, label in keys:
        fig.add_trace(go.Scatter(
            x=list(years),
            y=best["out"][k].tolist(),
            name=f"{label} — best fit",
            mode="lines",
            line=dict(dash="dash"),
            opacity=0.55,
            legendgroup=k,
            showlegend=True,
        ))
        fig.add_trace(go.Scatter(
            x=list(years),
            y=current["out"][k].tolist(),
            name=f"{label} — current",
            mode="lines+markers",
            line=dict(width=3),
            legendgroup=k,
            showlegend=True,
        ))
    fig.add_trace(go.Scatter(
        x=list(years),
        y=arr["sec_usgs"].tolist(),
        name="USGS secondary (target)",
        mode="markers",
        marker=dict(symbol="x", color="black", size=10),
        showlegend=True,
    ))
    fig.update_layout(
        title="Key flows: current vs. best fit",
        yaxis_title="t Pb / yr",
        xaxis_title="Year",
        height=420,
        margin=dict(l=50, r=20, t=50, b=40),
        legend=dict(font=dict(size=10)),
    )
    return fig


def render_india_calibration_explorer_tab(
    *,
    baci_df: pd.DataFrame,
    mining_df: pd.DataFrame,
    pb_factors: dict[int, float],
    active_years: list[int],
    time_period: str,
    mining_source: str,
    dataset_key: str,
) -> None:
    _mb_desc_col, _mb_lm_col = st.columns([8, 1])
    with _mb_desc_col:
        st.subheader("Mass Balance — India (Scenario B convention, τ=5 best fit)")
        st.write(
            "Fully reconciled material-flow model for India. Two independent "
            "observations anchor the system: USGS secondary smelting and a "
            "stock-derived install rate. Because they don't fully agree at "
            "the reported scale, three anchor methods are available — pick "
            "one below to see which assumption you're making. Adjust τ, "
            "k_stock, γ, β, and the η/δ efficiencies; φ_break_f and φ_smelt_f "
            "are always solved by the optimizer."
        )
    with _mb_lm_col:
        with st.popover("ℹ Learn more"):
            st.markdown(
                "**Process Estimates vs. Mass Balance:** the Process Estimates tab "
                "produces a single forward pass with fixed process parameters and "
                "covers many countries. **Mass Balance** additionally reconciles the "
                "outputs against external anchors — USGS primary/secondary refining "
                "and an external stock series — by running a constrained optimizer "
                "with smoothness regularization (λ=10). Currently India-only.\n\n"
                "**Three anchor methods**, all with the same parameter space and "
                "smoothness penalty, differ only in what is enforced exactly vs. "
                "minimized:\n\n"
                "- **Forward** — hard install equality (INSTALL_implied = "
                "INSTALL_target); minimize smelt residual. Treats trade-derived "
                "install as ground truth.\n"
                "- **Reversed** — hard smelt equality (SMELT_SEC = USGS); "
                "minimize install residual. Treats USGS as ground truth.\n"
                "- **Balanced** — no hard equality; both residuals enter the "
                "objective. Usually finds a joint Pareto-optimal point.\n\n"
                "If the two anchors agree, all three methods land at the same "
                "place. They don't — and the spread between them is the finding.\n\n"
                "**Sidebar adjusters apply here too:** BACI dataset, Pb content "
                "factors, and the Year + Time Period selection rebuild the inputs "
                "and recompute the best fits for all three anchors. Mining/refining "
                "source: India only has primary/secondary refining splits from "
                "USGS, so this tab uses USGS regardless of the sidebar selection."
            )

    # Fingerprint sidebar state and rebuild if it changed.
    fp = _fingerprint(dataset_key, pb_factors, active_years, time_period, mining_source)
    state = st.session_state.get(SESSION_KEY)
    if state is None or state.get("fingerprint") != fp:
        smooth_3yr = time_period.startswith("3-year")
        arr, build_warnings = build_arr_from_app(
            baci_df, mining_df, pb_factors, active_years, smooth_3yr=smooth_3yr
        )
        ctx = CalibrationContext(arr["year"].tolist())
        if ctx.n_y < 1:
            st.error("Calibration window is empty after clipping to available data.")
            return
        etas = dict(ETAS_DEFAULT)
        with st.spinner(f"Computing best fits for {list(ctx.years)} (τ pinned at 5)…"):
            best_by_anchor = _compute_all_best_fits(arr, etas, ctx)
        failed = [a for a, b in best_by_anchor.items() if not b.get("success")]
        if failed:
            st.error(f"Best-fit optimizer failed for: {', '.join(failed)}. "
                     f"Forward result: {best_by_anchor.get('forward', {}).get('message')}")
            return
        # Default to balanced -- gives the cleanest joint fit and avoids the
        # rails that forward/reversed hit when the anchors disagree.
        default_anchor = "balanced"
        state = _init_state_for(arr, etas, best_by_anchor, default_anchor, ctx, fp)
        st.session_state[SESSION_KEY] = state
        for key in list(st.session_state.keys()):
            if isinstance(key, str) and key.startswith("ice_"):
                del st.session_state[key]
        if build_warnings:
            for w in build_warnings:
                st.info(w)

    ctx: CalibrationContext = state["ctx"]
    arr = state["arr"]
    best = state["best"]
    years_tuple = ctx.years

    st.caption(
        f"**Calibration window:** {list(years_tuple)}  ·  "
        f"**Smoothing:** {'3-yr centered rolling mean' if time_period.startswith('3-year') else 'none (single year)'}  ·  "
        f"**BACI:** {dataset_key.upper()}  ·  "
        f"**Refining:** USGS (India-only constraint)"
    )

    if ctx.n_y == 1:
        st.info(
            "Single-year mode: only one year is being fit. The optimizer can match "
            "USGS secondary essentially exactly (MAE ≈ 0%); year-to-year drift "
            "regularization is inactive. Use **3-year average** in the sidebar for a "
            "more constrained fit."
        )

    # ── Anchor selector ──────────────────────────────────────────────────────
    current_anchor = state.get("anchor", "forward")
    sel_anchor = st.radio(
        "Anchor method",
        options=list(ANCHORS),
        format_func=lambda a: ANCHOR_LABELS[a],
        index=list(ANCHORS).index(current_anchor),
        horizontal=True,
        key="ice_anchor",
        help="Choose how to reconcile the two observations (USGS secondary "
             "smelting and the stock-derived install rate). Each method gives "
             "a different best fit; the spread between them is the finding.",
    )
    st.caption(ANCHOR_HELP[sel_anchor])
    if sel_anchor != current_anchor:
        _switch_anchor(state, sel_anchor)
        st.rerun()

    # ── Anchor comparison panel (all three best fits, current window) ────────
    with st.expander("📊 Compare best fits across anchor methods", expanded=False):
        ba = state["best_by_anchor"]
        comp_rows = []
        for a in ANCHORS:
            r = ba.get(a, {})
            if not r.get("success"):
                continue
            x = r["x"]
            comp_rows.append({
                "Anchor":          ANCHOR_LABELS[a],
                "τ":               round(float(x[ctx.index["tau"]]), 3),
                "k_stock":         round(float(x[ctx.index["k_stock"]]), 4),
                "MAE smelt (%)":   round(r["mae_smelt_pct"], 2),
                "MAE install (%)": round(r["mae_install_pct"], 2),
                "Max smelt gap (t Pb)":   round(r["max_smelt_gap_t_pb"], 0),
                "Max install gap (t Pb)": round(r["max_install_gap_t_pb"], 0),
                "J_total":         round(r["J_total"], 4),
            })
        if comp_rows:
            st.dataframe(pd.DataFrame(comp_rows), hide_index=True, use_container_width=True)
            st.caption(
                "Hard-equality residuals are exact (≈ 0) under their own anchor; "
                "they appear here as the *gap* the other anchor would see. The "
                "k_stock spread across rows brackets the calibrated effective stock."
            )
        else:
            st.warning("No successful best fits to compare.")

    # ── Year selector + lock-ratio toggle + action buttons ───────────────────
    ctl_a, ctl_b, ctl_c, ctl_d = st.columns([2, 2, 2, 2])
    with ctl_a:
        default_idx = max(0, len(years_tuple) - 2)
        sel_year = st.selectbox(
            "Year to edit",
            options=list(years_tuple),
            index=default_idx,
            key="ice_sel_year",
            help="γ and β sliders edit this year. Per-year readouts are for this year.",
        )
    with ctl_b:
        ratio_lock = st.toggle(
            "Lock k/τ ratio",
            value=state["ratio_lock"],
            key="ice_ratio_lock",
            help="When on, moving τ updates k_stock (and vice versa) to keep "
                 "k/τ constant. k/τ is proportional to the per-stock retirement rate.",
        )
        if ratio_lock != state["ratio_lock"]:
            state["ratio_lock"] = ratio_lock
            if ratio_lock:
                state["k_over_tau_locked"] = state["slider_k"] / state["slider_tau"]
    with ctl_c:
        if st.button("⟲ Reset to best fit", use_container_width=True, key="ice_reset_btn"):
            _reset_to_best(state)
            st.rerun()
    with ctl_d:
        refit_clicked = st.button(
            "▶ Re-fit",
            use_container_width=True,
            key="ice_refit_btn",
            type="primary",
            help="Hold the sliders fixed and re-optimize everything else.",
        )

    st.divider()

    # ── Sliders ──────────────────────────────────────────────────────────────
    col_scalars, col_tv = st.columns([1, 1])

    with col_scalars:
        st.markdown("**Scalar parameters**")
        lo, hi = BOUNDS["tau"]
        tau_val = st.slider(
            "τ — battery lifespan (yr)", float(lo), float(hi),
            float(state["slider_tau"]), 0.1, format="%.2f",
            key="ice_tau",
        )
        if state["ratio_lock"]:
            new_k = float(np.clip(state["k_over_tau_locked"] * tau_val,
                                  BOUNDS["k_stock"][0], BOUNDS["k_stock"][1]))
            state["slider_k"] = new_k
            st.session_state["ice_k"] = new_k
            st.caption(f"k_stock auto-set to {new_k:.4f} (k/τ = {state['k_over_tau_locked']:.4f})")
            k_val = new_k
            st.text_input("k_stock — stock scaling (locked)", value=f"{new_k:.4f}", disabled=True)
        else:
            lo, hi = BOUNDS["k_stock"]
            k_val = st.slider(
                "k_stock — stock scaling", float(lo), float(hi),
                float(state["slider_k"]), 0.01, format="%.3f",
                key="ice_k",
            )
        state["slider_tau"] = tau_val
        state["slider_k"]   = k_val

        st.markdown("**Physical efficiencies (apply to all years)**")
        eta_cols = st.columns(2)
        eta_names = [
            ("eta_break_f",  "η_break_F"),
            ("eta_break_i",  "η_break_I"),
            ("eta_scrap_f",  "η_scrap_F"),
            ("eta_scrap_i",  "η_scrap_I"),
            ("eta_mfg",      "η_mfg"),
            ("delta",        "δ (Pb at EoL)"),
        ]
        for i, (key, label) in enumerate(eta_names):
            with eta_cols[i % 2]:
                lo, hi = ETA_BOUNDS[key]
                v = st.slider(
                    label, float(lo), float(hi),
                    float(state["slider_etas"][key]), 0.01, format="%.2f",
                    key=f"ice_{key}",
                )
                state["slider_etas"][key] = v

    with col_tv:
        st.markdown(f"**Year-specific parameters — {sel_year}**")
        lo, hi = BOUNDS["gamma"]
        g_val = st.slider(
            "γ — collection rate", float(lo), float(hi),
            float(state["slider_gamma"][int(sel_year)]), 0.001, format="%.3f",
            key=f"ice_gamma_{int(sel_year)}",
        )
        lo, hi = BOUNDS["beta"]
        b_val = st.slider(
            "β — battery share of Pb demand", float(lo), float(hi),
            float(state["slider_beta"][int(sel_year)]), 0.005, format="%.3f",
            key=f"ice_beta_{int(sel_year)}",
        )
        state["slider_gamma"][int(sel_year)] = g_val
        state["slider_beta"][int(sel_year)]  = b_val

        if ctx.n_y > 1:
            st.markdown("<small>Other years (set by the optimizer):</small>", unsafe_allow_html=True)
            other_rows = []
            for y in years_tuple:
                yi = int(y)
                if yi == int(sel_year):
                    continue
                other_rows.append({
                    "Year": yi,
                    "γ":   round(state["slider_gamma"][yi], 4),
                    "β":   round(state["slider_beta"][yi], 4),
                })
            st.dataframe(pd.DataFrame(other_rows), hide_index=True, use_container_width=True)

        st.markdown("**Driven values for selected year (from current fit)**")
        cur_out = state["current"]["out"]
        ti = list(years_tuple).index(int(sel_year))
        drv_df = pd.DataFrame([{
            "φ_break_f": round(float(cur_out["phi_break_f"][ti]), 4),
            "φ_smelt_f": round(float(cur_out["phi_smelt_f"][ti]), 4),
        }])
        st.dataframe(drv_df, hide_index=True, use_container_width=True)

    if refit_clicked:
        with st.spinner("Re-fitting (SLSQP, 4 restarts)…"):
            _do_refit(state, int(sel_year))
        st.rerun()

    st.divider()

    # ── Readouts ─────────────────────────────────────────────────────────────
    cur = state["current"]
    anchor = state.get("anchor", "forward")
    st.markdown(f"### Fit quality — *{ANCHOR_LABELS[anchor]}*")
    m1, m2, m3, m4 = st.columns(4)
    delta_smelt = cur["mae_smelt_pct"] - best["mae_smelt_pct"]
    delta_inst  = cur["mae_install_pct"] - best["mae_install_pct"]
    delta_j     = cur["J_total"] - best["J_total"]

    # Help strings depend on which anchor is active: smelt is hard-constrained
    # in reversed, install is hard-constrained in forward, neither in balanced.
    if anchor == "forward":
        smelt_help = (f"Soft residual (being minimized). Best fit MAE_smelt = "
                      f"{best['mae_smelt_pct']:.2f}%.")
        inst_help  = "Hard constraint — should stay ≈ 0% if feasible."
        j_help     = (f"smelt_sse + smoothness. Best fit J_total = "
                      f"{best['J_total']:.4f}.")
        viol_label = "Max install eq. viol. (t Pb)"
        viol_help  = ("Magnitude of install equality-constraint violation "
                      "across years. Feasibility tolerance: 5,000 t Pb.")
    elif anchor == "reversed":
        smelt_help = "Hard constraint — should stay ≈ 0% if feasible."
        inst_help  = (f"Soft residual (being minimized). Best fit MAE_install = "
                      f"{best['mae_install_pct']:.2f}%.")
        j_help     = (f"install_sse + smoothness. Best fit J_total = "
                      f"{best['J_total']:.4f}.")
        viol_label = "Max smelt eq. viol. (t Pb)"
        viol_help  = ("Magnitude of smelt equality-constraint violation "
                      "across years. Feasibility tolerance: 5,000 t Pb.")
    else:  # balanced
        smelt_help = (f"Soft residual (being minimized). Best fit MAE_smelt = "
                      f"{best['mae_smelt_pct']:.2f}%.")
        inst_help  = (f"Soft residual (being minimized). Best fit MAE_install = "
                      f"{best['mae_install_pct']:.2f}%.")
        j_help     = (f"w_s*smelt_sse + w_i*install_sse + smoothness. "
                      f"Best fit J_total = {best['J_total']:.4f}.")
        viol_label = "Max gap, larger of (t Pb)"
        viol_help  = ("Largest of (smelt gap, install gap). No hard constraint "
                      "in balanced mode — both gaps are soft residuals.")

    m1.metric(
        "MAE smelt (%)",
        f"{cur['mae_smelt_pct']:.2f}%",
        delta=f"{delta_smelt:+.2f} vs best",
        delta_color="inverse",
        help=smelt_help,
    )
    m2.metric(
        "MAE install (%)",
        f"{cur['mae_install_pct']:.3f}%",
        delta=f"{delta_inst:+.3f} vs best",
        delta_color="inverse",
        help=inst_help,
    )
    m3.metric(
        "J_total",
        f"{cur['J_total']:.4f}",
        delta=f"{delta_j:+.4f} vs best",
        delta_color="inverse",
        help=j_help,
    )
    # For balanced, show the larger of the two soft gaps. Otherwise show the
    # anchor's equality-constraint violation directly.
    if anchor == "balanced":
        viol_value = max(cur["max_smelt_gap_t_pb"], cur["max_install_gap_t_pb"])
    else:
        viol_value = cur["max_eq_violation_t_pb"]
    m4.metric(
        viol_label,
        f"{viol_value:,.0f}",
        help=viol_help,
    )

    ti = list(years_tuple).index(int(sel_year))
    res_smelt_pct = float(cur["out"]["res_smelt_pct"][ti] * 100)
    res_inst_pct  = float(cur["out"]["res_install_pct"][ti] * 100)
    st.markdown(f"**{sel_year} residuals:** "
                f"smelt = `{res_smelt_pct:+.2f}%`  ·  install = `{res_inst_pct:+.3f}%`")

    if not cur.get("feasible", True) and anchor in ("forward", "reversed"):
        which = "install" if anchor == "forward" else "smelt"
        st.warning(
            f"Current fit is **infeasible**: max {which}-equality violation = "
            f"{cur['max_eq_violation_t_pb']:,.0f} t Pb (> 5,000 t Pb tolerance). "
            "Try relaxing pins or restoring the best fit."
        )

    if anchor == "forward":
        smelt_chart_title = "Smelt residual (soft, being minimized)"
        inst_chart_title  = "Install residual (hard-constrained to ≈ 0)"
    elif anchor == "reversed":
        smelt_chart_title = "Smelt residual (hard-constrained to ≈ 0)"
        inst_chart_title  = "Install residual (soft, being minimized)"
    else:
        smelt_chart_title = "Smelt residual (soft, being minimized)"
        inst_chart_title  = "Install residual (soft, being minimized)"

    chart_a, chart_b = st.columns(2)
    with chart_a:
        st.plotly_chart(
            _residual_chart(cur, best, "res_smelt_pct",
                            smelt_chart_title,
                            "% of USGS secondary", years_tuple),
            use_container_width=True,
        )
    with chart_b:
        st.plotly_chart(
            _residual_chart(cur, best, "res_install_pct",
                            inst_chart_title,
                            "% of INSTALL_target", years_tuple),
            use_container_width=True,
        )

    st.plotly_chart(_flow_chart(cur, best, years_tuple, arr), use_container_width=True)

    with st.expander("Full parameter table (current vs. best fit)", expanded=False):
        rows = []
        x_cur = cur.get("x")
        x_best = best["x"]
        for label, key in [("τ", "tau"), ("k_stock", "k_stock")]:
            rows.append({
                "Parameter": label, "Year": "",
                "Current": round(float(x_cur[ctx.index[key]]) if x_cur is not None else float("nan"), 4),
                "Best fit": round(float(x_best[ctx.index[key]]), 4),
            })
        for name in ("gamma", "beta", "phi_break_f", "phi_smelt_f"):
            for y in years_tuple:
                yi = int(y)
                rows.append({
                    "Parameter": name,
                    "Year": yi,
                    "Current": round(float(x_cur[ctx.tv_slot(name, yi)]) if x_cur is not None else float("nan"), 4),
                    "Best fit": round(float(x_best[ctx.tv_slot(name, yi)]), 4),
                })
        for k, label in [
            ("delta", "δ"),
            ("eta_break_f", "η_break_F"),
            ("eta_break_i", "η_break_I"),
            ("eta_scrap_f", "η_scrap_F"),
            ("eta_scrap_i", "η_scrap_I"),
            ("eta_mfg", "η_mfg"),
        ]:
            rows.append({
                "Parameter": label, "Year": "",
                "Current": round(state["slider_etas"][k], 3),
                "Best fit": round(state["etas_best"][k], 3),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
