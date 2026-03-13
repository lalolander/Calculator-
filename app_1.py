"""
Decompression Teaching Tool v3
===============================
Focuses exclusively on two concepts:
  1. Off-gassing Start Depth  — the depth at which any tissue becomes supersaturated
  2. Ceiling Depth            — the minimum safe depth, per Bühlmann ZHL-16C and VPM-B

No deco stop scheduling. No total deco time. Just the physics of those two depths.
"""

import numpy as np
import streamlit as st
import math
import pandas as pd

# =============================================================================
# CONFIGURATION — ZHL-16C Coefficients (Bühlmann 1995, Baker 1998)
# =============================================================================

class DiveConfig:
    N_COMPARTMENTS = 16

    # Half-times (minutes)
    HALF_TIMES_N2 = np.array([
        4.0, 8.0, 12.5, 18.5, 27.0, 38.3, 54.3, 77.0,
        109.0, 146.0, 187.0, 239.0, 305.0, 390.0, 498.0, 635.0
    ])
    HALF_TIMES_HE = np.array([
        1.51, 3.02, 4.72, 6.99, 10.21, 14.48, 20.53, 29.11,
        41.20, 55.19, 70.69, 90.34, 115.29, 147.42, 188.24, 240.03
    ])

    # ZHL-16C 'a' coefficients (bar) — N2
    A_N2 = np.array([
        1.2599, 1.0000, 0.8618, 0.7562, 0.6200, 0.5043, 0.4410, 0.4000,
        0.3750, 0.3500, 0.3295, 0.3065, 0.2835, 0.2610, 0.2480, 0.2327
    ])
    # ZHL-16C 'b' coefficients (dimensionless) — N2
    B_N2 = np.array([
        0.5050, 0.6514, 0.7222, 0.7825, 0.8126, 0.8434, 0.8693, 0.8910,
        0.9092, 0.9222, 0.9319, 0.9403, 0.9477, 0.9544, 0.9602, 0.9653
    ])
    # ZHL-16C 'a' coefficients (bar) — He
    A_HE = np.array([
        1.7424, 1.3830, 1.1919, 1.0458, 0.9220, 0.8205, 0.7305, 0.6502,
        0.5950, 0.5545, 0.5333, 0.5189, 0.5181, 0.5176, 0.5172, 0.5119
    ])
    # ZHL-16C 'b' coefficients (dimensionless) — He
    B_HE = np.array([
        0.4245, 0.5747, 0.6527, 0.7223, 0.7582, 0.7957, 0.8279, 0.8553,
        0.8757, 0.8903, 0.8997, 0.9073, 0.9122, 0.9171, 0.9217, 0.9267
    ])

    # Environment
    SURFACE_PRESSURE    = 1.013    # bar (1 atm)
    BAR_PER_METER       = 0.09985  # bar/m seawater (1025 kg/m³)
    WATER_VAPOUR_PRESS  = 0.0627   # bar at 37 °C (alveolar)

    # Ascent rate for the scan-only simulation
    ASCENT_RATE  = 9.0   # m/min
    DESCENT_RATE = 20.0  # m/min

    # VPM-B nucleus parameters (Yount 1979)
    GAMMA_C   = 0.0179   # critical surface tension, N/m
    R_INITIAL = 0.8e-6   # initial bubble nucleus radius, m (0.8 μm)
    LAMBDA_CE = 7500.0   # skin compression coefficient, Pa/bar equivalent

    # Pre-computed rate constants
    k_n2 = np.log(2) / HALF_TIMES_N2
    k_he = np.log(2) / HALF_TIMES_HE

    @classmethod
    def mixed_ab(cls, fn2: float, fhe: float):
        """
        He-fraction-weighted interpolation of a and b (Baker/Bühlmann method).
        Used for Trimix where both N2 and He contribute to tissue loading.
        """
        denom = fn2 + fhe
        if denom < 1e-9:
            return cls.A_N2.copy(), cls.B_N2.copy()
        a = (cls.A_N2 * fn2 + cls.A_HE * fhe) / denom
        b = (cls.B_N2 * fn2 + cls.B_HE * fhe) / denom
        return a, b


cfg = DiveConfig()


# =============================================================================
# PHYSICS FUNCTIONS
# =============================================================================

def depth_to_pressure(depth: float) -> float:
    """Absolute pressure at depth (bar)."""
    return cfg.SURFACE_PRESSURE + depth * cfg.BAR_PER_METER


def pressure_to_depth(p: float) -> float:
    """Convert absolute pressure back to depth (m), clipped to 0."""
    return max(0.0, (p - cfg.SURFACE_PRESSURE) / cfg.BAR_PER_METER)


def inspired_pp(depth: float, fo2: float, fn2: float, fhe: float):
    """
    Alveolar inspired partial pressures, corrected for water vapour.
    P_insp_gas = fgas * (P_amb - P_H2O)
    """
    p_amb = depth_to_pressure(depth)
    p_dry = max(0.0, p_amb - cfg.WATER_VAPOUR_PRESS)
    return fo2 * p_dry, fn2 * p_dry, fhe * p_dry


def schreiner_load(depth: float, t: float,
                   fo2: float, fn2: float, fhe: float,
                   n2_0: np.ndarray, he_0: np.ndarray):
    """
    Exact Haldanian exponential gas loading for a constant-depth segment.
    Valid for on-gassing and off-gassing equally — direction is automatic.

    P_t = P_insp + (P_0 - P_insp) * exp(-k * t)
    """
    _, p_n2, p_he = inspired_pp(depth, fo2, fn2, fhe)
    n2 = p_n2 + (n2_0 - p_n2) * np.exp(-cfg.k_n2 * t)
    he = p_he + (he_0 - p_he) * np.exp(-cfg.k_he * t)
    return n2, he


# --- Bühlmann Ceiling ---

def buhlmann_ceiling_depth(n2: np.ndarray, he: np.ndarray,
                            fn2: float, fhe: float, gf: float = 1.0) -> tuple:
    """
    Minimum ambient pressure (depth) at which NO tissue exceeds its M-value.

    M-value line:  M = P_t / b + a   (Bühlmann)
    Rearranged to minimum tolerated ambient pressure:
        P_min = (P_t - a * gf) * b

    Returns (ceiling_depth_m, leading_tissue_index).
    GF < 1.0 applies a gradient factor (Baker 1998) — more conservative.
    """
    a, b = cfg.mixed_ab(fn2, fhe)
    pt = n2 + he
    p_min = (pt - a * gf) * b          # minimum ambient pressure per compartment
    p_min = np.maximum(p_min, 0.0)     # can't be negative
    idx   = int(np.argmax(p_min))
    depth = pressure_to_depth(p_min[idx] + cfg.SURFACE_PRESSURE)
    return depth, idx


# --- VPM-B Ceiling ---

def vpm_crush_radius(max_pressure: float) -> np.ndarray:
    """
    After exposure to max_pressure (bar absolute), nuclei are compressed.
    All 16 compartments share the same crushing because VPM-B applies a
    single ambient-pressure history, not per-compartment nucleation.

    r_crit = (2 * gamma_c * r0) / (2 * gamma_c + lambda * delta_P)

    delta_P = max_pressure - surface_pressure  (the actual crush overpressure)
    """
    delta_p = max(0.0, max_pressure - cfg.SURFACE_PRESSURE)  # bar
    # Convert lambda from Pa/bar-equiv to consistent units (both in bar-space):
    # LAMBDA_CE is given in Pa-equivalent per bar; 1 bar = 100000 Pa
    # We keep everything in N/m and bar by noting:
    #   2*gamma_c has units N/m = J/m² → use directly
    #   lambda * delta_P must also be N/m → LAMBDA_CE is in N/m per bar of delta_P
    lam_bar = cfg.LAMBDA_CE / 1e5      # convert to (N/m)/bar
    numerator   = 2.0 * cfg.GAMMA_C * cfg.R_INITIAL
    denominator = 2.0 * cfg.GAMMA_C + lam_bar * delta_p
    r_crit = numerator / denominator
    return np.full(cfg.N_COMPARTMENTS, r_crit)   # same radius for all compartments


def vpm_allowable_supersaturation(r_crit: np.ndarray) -> np.ndarray:
    """
    Per-compartment allowable supersaturation (bar) before bubble nucleation.
    Young-Laplace:  delta_P_allow = 2 * gamma_c / r_crit
    """
    return (2.0 * cfg.GAMMA_C) / r_crit


def vpm_ceiling_depth(n2: np.ndarray, he: np.ndarray, r_crit: np.ndarray) -> tuple:
    """
    VPM-B ceiling: the depth above which bubble nucleation would occur.
    Per compartment:
        P_min_amb = P_tissue - delta_P_allow
    Returns (ceiling_depth_m, leading_tissue_index).
    """
    delta_p_allow = vpm_allowable_supersaturation(r_crit)
    p_min = (n2 + he) - delta_p_allow
    p_min = np.maximum(p_min, 0.0)
    idx   = int(np.argmax(p_min))
    depth = pressure_to_depth(p_min[idx] + cfg.SURFACE_PRESSURE)
    return depth, idx


# --- Off-gassing Start ---

def offgas_start_depth(n2: np.ndarray, he: np.ndarray) -> tuple:
    """
    The depth at which the diver is currently sitting where ANY tissue
    pressure exceeds ambient — the first moment tissues start releasing gas.

    Rearranged: P_tissue > P_amb  →  depth < (P_tissue - P_surface) / BAR_PER_METER
    The shallowest depth at which this happens for the *most loaded* tissue
    defines where off-gassing begins.

    Returns (offgas_depth_m, leading_tissue_index).
    If tissues are below ambient (still on-gassing), returns (0.0, -1).
    """
    # For each tissue, the depth at which it exactly equals ambient:
    # P_t = P_surface + depth * BAR_PER_METER  →  depth = (P_t - P_surface) / BAR_PER_METER
    depths = pressure_to_depth(n2 + he)        # works element-wise (returns array)
    # "Off-gassing start" = the *deepest* of these depths, because that tissue
    # requires the highest ambient to stay in solution.
    idx   = int(np.argmax(depths))
    depth = float(depths[idx])
    if depth <= 0.0:
        return 0.0, -1
    return depth, idx


def round_up_3m(d: float) -> float:
    """Round a depth UP to the next 3 m increment."""
    if d <= 0:
        return 0.0
    return math.ceil(d / 3.0) * 3.0


# =============================================================================
# SIMULATION — Descent + Bottom Only (no stop scheduling)
# =============================================================================

def simulate_bottom(max_depth: float, bottom_time: float,
                    fo2: float, fn2: float, fhe: float,
                    dt: float = 0.25) -> tuple:
    """
    Simulates descent then bottom phase with continuous tissue loading.
    Returns (n2_tissues, he_tissues, r_crit, max_ambient_pressure).

    dt : time step in minutes (0.25 min = 15 s)
    """
    # Surface equilibrium (79.02% N2 at surface dry pressure)
    p_dry_surface = cfg.SURFACE_PRESSURE - cfg.WATER_VAPOUR_PRESS
    n2 = np.full(cfg.N_COMPARTMENTS, 0.7902 * p_dry_surface)
    he = np.zeros(cfg.N_COMPARTMENTS)

    max_amb = cfg.SURFACE_PRESSURE   # tracks peak pressure for VPM-B crushing

    # --- Descent ---
    descent_time  = max_depth / cfg.DESCENT_RATE
    steps_descent = max(1, int(descent_time / dt))
    for i in range(steps_descent):
        frac  = (i + 1) / steps_descent
        depth = frac * max_depth
        n2, he = schreiner_load(depth, dt, fo2, fn2, fhe, n2, he)
        max_amb = max(max_amb, depth_to_pressure(depth))

    # --- Bottom ---
    steps_bottom = max(1, int(bottom_time / dt))
    for _ in range(steps_bottom):
        n2, he = schreiner_load(max_depth, dt, fo2, fn2, fhe, n2, he)

    max_amb = max(max_amb, depth_to_pressure(max_depth))

    # Compute VPM-B critical radii based on maximum pressure seen
    r_crit = vpm_crush_radius(max_amb)

    return n2, he, r_crit, max_amb


# =============================================================================
# STREAMLIT UI
# =============================================================================

st.set_page_config(
    page_title="Deco Teaching Tool",
    page_icon="🫧",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
body { background-color: #0b1320; }

.kpi-card {
    background: #101e30;
    border: 1px solid #1c3a5c;
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
    margin-bottom: 4px;
}
.kpi-label  { font-size: 0.72rem; color: #5b8fb9; text-transform: uppercase;
               letter-spacing: 0.1em; margin-bottom: 4px; }
.kpi-value  { font-size: 2rem; font-weight: 700; color: #e8f4ff; line-height: 1.1; }
.kpi-sub    { font-size: 0.72rem; color: #6a99bf; margin-top: 4px; }
.kpi-warn   { border-color: #8b4914; }
.kpi-ok     { border-color: #1a7a3a; }

.model-row  {
    display: flex; gap: 12px; margin-bottom: 10px;
}
.model-card {
    flex: 1;
    background: #0d1b2a;
    border-radius: 8px;
    border-left: 4px solid #1a6eb5;
    padding: 12px 16px;
}
.model-card.vpm { border-left-color: #b56a1a; }
.model-title { font-size: 0.75rem; color: #5b8fb9; text-transform: uppercase;
               letter-spacing: 0.08em; margin-bottom: 6px; }
.model-depth { font-size: 1.5rem; font-weight: 700; color: #e0f0ff; }
.model-meta  { font-size: 0.72rem; color: #7a9bbf; margin-top: 4px; }

.concept-box {
    background: #0d1e2e;
    border: 1px solid #1c3a5c;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 8px 0;
    font-size: 0.88rem;
    color: #c0d8f0;
    line-height: 1.6;
}
.concept-box h4 { color: #7ab8e0; margin: 0 0 6px 0; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Dive Profile")

    fo2 = st.slider("Oxygen (%)", 5, 100, 21) / 100.0
    fhe = st.slider("Helium (%)",  0,  95, 35) / 100.0
    fn2 = round(1.0 - fo2 - fhe, 6)

    if fn2 < -0.001:
        st.error("❌ O₂ + He > 100%")
        st.stop()
    fn2 = max(fn2, 0.0)

    ca, cb, cc = st.columns(3)
    ca.metric("O₂", f"{fo2*100:.0f}%")
    cb.metric("He", f"{fhe*100:.0f}%")
    cc.metric("N₂", f"{fn2*100:.0f}%")

    st.divider()
    max_depth   = st.number_input("Max Depth (m)",       1.0, 330.0,  40.0, 1.0)
    bottom_time = st.number_input("Bottom Time (min)",   1.0, 480.0,  35.0, 1.0)

    st.divider()
    gf = st.slider(
        "Bühlmann GF (conservatism)",
        min_value=0.50, max_value=1.00, value=1.00, step=0.05,
        help="1.0 = pure M-value limit. 0.80 = use only 80% of the headroom (more conservative)."
    )

    st.divider()
    calc_btn = st.button("🔄 Calculate", use_container_width=True, type="primary")

# ── Calculation ───────────────────────────────────────────────────────────────
if calc_btn or "result" not in st.session_state:
    n2, he, r_crit, max_amb = simulate_bottom(max_depth, bottom_time, fo2, fn2, fhe)

    offgas_depth,  offgas_idx  = offgas_start_depth(n2, he)
    bhl_ceil,      bhl_idx     = buhlmann_ceiling_depth(n2, he, fn2, fhe, gf=gf)
    vpm_ceil,      vpm_idx     = vpm_ceiling_depth(n2, he, r_crit)

    # Practical 3 m rounded values
    offgas_3m = round_up_3m(offgas_depth)
    bhl_3m    = round_up_3m(bhl_ceil)
    vpm_3m    = round_up_3m(vpm_ceil)

    # Safety margin: gap between off-gassing start and ceiling
    margin_bhl = max(0.0, offgas_depth - bhl_ceil)
    margin_vpm = max(0.0, offgas_depth - vpm_ceil)

    st.session_state["result"] = dict(
        n2=n2, he=he, r_crit=r_crit, max_amb=max_amb,
        offgas_depth=offgas_depth, offgas_idx=offgas_idx,
        bhl_ceil=bhl_ceil, bhl_idx=bhl_idx,
        vpm_ceil=vpm_ceil, vpm_idx=vpm_idx,
        offgas_3m=offgas_3m, bhl_3m=bhl_3m, vpm_3m=vpm_3m,
        margin_bhl=margin_bhl, margin_vpm=margin_vpm,
        gf=gf, fn2=fn2, fhe=fhe,
    )

r = st.session_state["result"]

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("🫧 Decompression Teaching Tool")
st.caption("Focused on two concepts only: **Off-gassing Start Depth** and **Ceiling Depth** "
           "— calculated from Bühlmann ZHL-16C and VPM-B independently.")

st.divider()

# ── KPI Row ───────────────────────────────────────────────────────────────────
k1, k2, k3 = st.columns(3)

with k1:
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">Off-gassing Start</div>
      <div class="kpi-value">{r['offgas_depth']:.1f} m</div>
      <div class="kpi-sub">
        Practical (3 m increment): <strong>{r['offgas_3m']:.0f} m</strong><br>
        Leading tissue: T{r['offgas_idx']+1} 
        (ht&nbsp;{cfg.HALF_TIMES_N2[r['offgas_idx']]:.0f} min N₂)
      </div>
    </div>""", unsafe_allow_html=True)

with k2:
    card_cls = "kpi-card kpi-warn" if r['bhl_ceil'] > 0 else "kpi-card kpi-ok"
    st.markdown(f"""
    <div class="{card_cls}">
      <div class="kpi-label">Bühlmann Ceiling (GF {r['gf']:.2f})</div>
      <div class="kpi-value">{r['bhl_ceil']:.1f} m</div>
      <div class="kpi-sub">
        Practical: <strong>{r['bhl_3m']:.0f} m</strong> &nbsp;|&nbsp;
        Margin: {r['margin_bhl']:.1f} m<br>
        Leading tissue: T{r['bhl_idx']+1}
        (ht&nbsp;{cfg.HALF_TIMES_N2[r['bhl_idx']]:.0f} min N₂)
      </div>
    </div>""", unsafe_allow_html=True)

with k3:
    card_cls = "kpi-card kpi-warn" if r['vpm_ceil'] > 0 else "kpi-card kpi-ok"
    st.markdown(f"""
    <div class="{card_cls}">
      <div class="kpi-label">VPM-B Ceiling</div>
      <div class="kpi-value">{r['vpm_ceil']:.1f} m</div>
      <div class="kpi-sub">
        Practical: <strong>{r['vpm_3m']:.0f} m</strong> &nbsp;|&nbsp;
        Margin: {r['margin_vpm']:.1f} m<br>
        Leading tissue: T{r['vpm_idx']+1}
        (ht&nbsp;{cfg.HALF_TIMES_N2[r['vpm_idx']]:.0f} min N₂)
      </div>
    </div>""", unsafe_allow_html=True)

st.divider()

# ── Depth Zone Diagram ────────────────────────────────────────────────────────
st.subheader("Depth Zone Diagram")

zone_df = pd.DataFrame({
    "Zone": [
        "Surface",
        f"Bühlmann Ceiling ({r['bhl_ceil']:.1f} m)",
        f"VPM-B Ceiling ({r['vpm_ceil']:.1f} m)",
        f"Off-gassing Start ({r['offgas_depth']:.1f} m)",
        f"Bottom ({max_depth:.0f} m)",
    ],
    "Depth (m)": [
        0.0,
        r['bhl_ceil'],
        r['vpm_ceil'],
        r['offgas_depth'],
        max_depth,
    ]
})
st.bar_chart(zone_df, x="Zone", y="Depth (m)", horizontal=True, height=280)

st.caption(
    "**Off-gassing Start** is where tissues first become supersaturated relative to ambient. "
    "**Ceilings** are the depths above which tissue pressures exceed tolerated limits (Bühlmann M-value "
    "or VPM-B bubble nucleation threshold). The gap between them is the safe ascent window."
)

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🫁 Tissue Loading",
    "🔬 VPM-B Bubble State",
    "📖 Physics Explainer",
])

# ─── Tab 1: Tissue Loading ────────────────────────────────────────────────────
with tab1:
    n2, he = r["n2"], r["he"]
    fn2_, fhe_ = r["fn2"], r["fhe"]
    a, b = cfg.mixed_ab(fn2_, fhe_)

    total = n2 + he

    # M-value at the current ceiling depth (Bühlmann)
    p_ceil_abs = cfg.SURFACE_PRESSURE + r["bhl_ceil"] * cfg.BAR_PER_METER
    # M-value at surface (GF=1) for reference
    mv_surface = a + cfg.SURFACE_PRESSURE / b

    # Supersaturation relative to surface M-value
    sat_pct = np.clip(total / mv_surface * 100, 0, 200)

    # Per-compartment Bühlmann ceiling
    p_comp_ceil = np.maximum((total - a * r["gf"]) * b, 0.0)
    comp_ceil_depth = np.array([pressure_to_depth(p + cfg.SURFACE_PRESSURE) for p in p_comp_ceil])

    # Per-compartment VPM allowable supersaturation
    vpm_allow = vpm_allowable_supersaturation(r["r_crit"])

    tissue_df = pd.DataFrame({
        "Tissue":             [f"T{i+1}" for i in range(cfg.N_COMPARTMENTS)],
        "N₂ Half-time (min)": cfg.HALF_TIMES_N2,
        "He Half-time (min)": cfg.HALF_TIMES_HE,
        "N₂ Load (bar)":      np.round(n2, 4),
        "He Load (bar)":      np.round(he, 4),
        "Total Inert (bar)":  np.round(total, 4),
        "M-val Limit @ surf (bar)": np.round(mv_surface, 4),
        "Saturation %":       np.round(sat_pct, 1),
        "Bühlmann Ceil Depth (m)":  np.round(comp_ceil_depth, 2),
        "VPM Allow ΔP (bar)":       np.round(vpm_allow, 4),
    })

    st.dataframe(tissue_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Total Inert Load vs Bühlmann M-value Limit (surface)")

    chart_df = pd.DataFrame({
        "Tissue":               [f"T{i+1}" for i in range(cfg.N_COMPARTMENTS)],
        "Total Inert Gas (bar)": np.round(total, 4),
        "M-value Limit (bar)":   np.round(mv_surface, 4),
    })
    st.bar_chart(chart_df, x="Tissue", y=["Total Inert Gas (bar)", "M-value Limit (bar)"])
    st.caption(
        "Bars above the M-value limit indicate tissues that require depth to stay in solution. "
        "The most violated compartment drives the ceiling."
    )

    st.divider()
    st.subheader("Per-Compartment Bühlmann Ceiling Depth")

    ceil_df = pd.DataFrame({
        "Tissue":              [f"T{i+1}" for i in range(cfg.N_COMPARTMENTS)],
        "Ceiling Depth (m)":   np.round(comp_ceil_depth, 2),
    })
    st.bar_chart(ceil_df, x="Tissue", y="Ceiling Depth (m)")
    st.caption(
        "Each bar shows the minimum depth (m) that compartment requires. "
        f"The overall Bühlmann ceiling is the maximum: **{r['bhl_ceil']:.1f} m** (T{r['bhl_idx']+1})."
    )


# ─── Tab 2: VPM-B Bubble State ───────────────────────────────────────────────
with tab2:
    st.subheader("VPM-B: Bubble Nucleus State After Bottom Phase")

    r_crit  = r["r_crit"]
    r_init  = np.full(cfg.N_COMPARTMENTS, cfg.R_INITIAL)
    delta_p_allow = vpm_allowable_supersaturation(r_crit)

    # Per-compartment VPM ceiling
    pt = r["n2"] + r["he"]
    p_min_vpm = np.maximum(pt - delta_p_allow, 0.0)
    vpm_comp_depths = np.array([pressure_to_depth(p + cfg.SURFACE_PRESSURE) for p in p_min_vpm])

    crush_delta = max_depth * cfg.BAR_PER_METER   # bar over surface
    r_crit_scalar = float(r_crit[0])              # same for all compartments

    col1, col2, col3 = st.columns(3)
    col1.metric("Initial r₀", f"{cfg.R_INITIAL*1e6:.3f} μm")
    col2.metric("r_crit after dive", f"{r_crit_scalar*1e6:.4f} μm",
                delta=f"{(r_crit_scalar - cfg.R_INITIAL)*1e6:.4f} μm",
                delta_color="normal")
    col3.metric("Allowable ΔP (bar)", f"{delta_p_allow[0]:.4f}")

    st.caption(
        f"Crush overpressure at {max_depth:.0f} m: **{crush_delta:.3f} bar**. "
        f"Nuclei compressed from {cfg.R_INITIAL*1e6:.3f} μm → {r_crit_scalar*1e6:.4f} μm. "
        f"Smaller radius = harder to nucleate = higher allowable supersaturation."
    )

    st.divider()

    vpm_df = pd.DataFrame({
        "Tissue":                     [f"T{i+1}" for i in range(cfg.N_COMPARTMENTS)],
        "Total Inert (bar)":           np.round(pt, 4),
        "Allowable ΔP (bar)":          np.round(delta_p_allow, 4),
        "VPM Ceiling Depth (m)":       np.round(vpm_comp_depths, 2),
    })
    st.dataframe(vpm_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Per-Compartment VPM-B Ceiling Depth")

    vpm_ceil_df = pd.DataFrame({
        "Tissue":             [f"T{i+1}" for i in range(cfg.N_COMPARTMENTS)],
        "VPM Ceiling (m)":    np.round(vpm_comp_depths, 2),
    })
    st.bar_chart(vpm_ceil_df, x="Tissue", y="VPM Ceiling (m)")
    st.caption(
        "Each bar is the depth required by the VPM-B bubble criterion for that compartment. "
        f"Overall VPM-B ceiling: **{r['vpm_ceil']:.1f} m** (T{r['vpm_idx']+1})."
    )

    st.divider()
    st.subheader("Total Inert Load vs VPM-B Allowable Supersaturation")

    allow_chart = pd.DataFrame({
        "Tissue":              [f"T{i+1}" for i in range(cfg.N_COMPARTMENTS)],
        "Total Inert (bar)":   np.round(pt, 4),
        "Allowable ΔP (bar)":  np.round(delta_p_allow, 4),
    })
    st.bar_chart(allow_chart, x="Tissue", y=["Total Inert (bar)", "Allowable ΔP (bar)"])
    st.caption(
        "Where 'Total Inert' exceeds 'Allowable ΔP', those compartments would nucleate "
        "bubbles if the diver were at the surface."
    )


# ─── Tab 3: Physics Explainer ─────────────────────────────────────────────────
with tab3:
    st.subheader("📖 What This Tool Calculates and Why")

    with st.expander("1 — Off-gassing Start Depth", expanded=True):
        st.markdown(r"""
**Definition**: The depth at which the most-loaded tissue compartment first has a partial
pressure of inert gas **greater than the ambient pressure**.

At this point, the gradient reverses: gas starts moving *out of* tissue into blood and lungs.
Above this depth, you are off-gassing. Below it, you are still on-gassing.

$$P_{tissue} > P_{ambient} \implies \text{off-gassing}$$

Per compartment, the off-gassing starts at depth:

$$d_{offgas,i} = \frac{P_{t,i} - P_{surface}}{0.09985 \text{ bar/m}}$$

The **overall off-gassing start depth** is the maximum across all 16 compartments — it tells
you the deepest point at which any tissue begins to release gas.

> **Teaching point**: Off-gassing *starting* doesn't mean it is safe. The rate of release
> matters. If you ascend too quickly above the ceiling, the release becomes uncontrolled → bubbles.
        """)

    with st.expander("2 — Bühlmann ZHL-16C Ceiling", expanded=True):
        st.markdown(r"""
**Bühlmann M-value**: Each compartment has a maximum tolerated inert gas pressure that
depends linearly on ambient pressure:

$$M = \frac{P_{amb}}{b} + a$$

Rearranged to find the **minimum safe ambient pressure** (ceiling):

$$P_{ceil} = (P_{tissue} - a \cdot GF) \cdot b$$

where:
| Symbol | Meaning |
|--------|---------|
| $a$ | Y-intercept coefficient — controls max supersaturation at surface |
| $b$ | Slope coefficient — controls how limit grows with depth |
| $GF$ | Gradient Factor (0–1) — scales conservatism; 1.0 = exact M-value |

For **Trimix**, $a$ and $b$ are He-fraction weighted:
$$a_{mix} = \frac{a_{N_2} \cdot f_{N_2} + a_{He} \cdot f_{He}}{f_{N_2} + f_{He}}$$

The ceiling depth is:
$$d_{ceil,Bhl} = \max_i \left[\frac{P_{ceil,i} - P_{surface}}{0.09985}\right]$$

> **Teaching point**: The Bühlmann model does not care about bubble *size* — only whether
> supersaturation exceeds the empirically derived M-value line. GF is a pragmatic fudge factor
> to add headroom that M-values alone don't provide.
        """)

    with st.expander("3 — VPM-B Ceiling (Bubble Nucleation)", expanded=True):
        st.markdown(r"""
**VPM = Varying Permeability Model** (Yount 1979, Yount & Hoffman 1986).

Bubble nuclei are stabilised by a surfactant skin with surface tension $\gamma_c$.
A nucleus of radius $r$ is in equilibrium when:

$$\Delta P_{allow} = \frac{2\gamma_c}{r}$$

**Crushing effect**: Descent to depth $d$ imposes overpressure $\Delta P_{crush}$,
which compresses nuclei from initial radius $r_0$ to a smaller $r_{crit}$:

$$r_{crit} = \frac{2\gamma_c \cdot r_0}{2\gamma_c + \lambda \cdot \Delta P_{crush}}$$

Smaller $r_{crit}$ → larger $\Delta P_{allow}$ → nuclei are **harder to grow** (protective crushing).

**VPM-B ceiling per compartment**:
$$P_{min,i} = P_{t,i} - \Delta P_{allow}$$
$$d_{ceil,VPM} = \max_i \left[\frac{P_{min,i} - P_{surface}}{0.09985}\right]$$

> **Teaching point**: VPM-B explicitly models bubble *physics*, so it naturally produces
> deeper first stops than Bühlmann (the crushing effect means you need more ambient pressure
> to keep small nuclei stable). It's history-dependent — previous dives change $r_{crit}$.

| Parameter | Value used |
|-----------|-----------|
| $\gamma_c$ | 0.0179 N/m |
| $r_0$ | 0.8 μm |
| $\lambda$ | 7500 Pa·m/N (converted) |
        """)

    with st.expander("4 — Why the Two Models Give Different Ceilings", expanded=False):
        st.markdown(r"""
**Bühlmann** is empirical: the $a$ and $b$ coefficients were derived from experimental
dives and decompression sickness outcome data. It asks: *"Have any known-safe limits been exceeded?"*

**VPM-B** is mechanistic: it models the physical stability of bubble nuclei using
Young-Laplace surface tension equations. It asks: *"Would the pressure differential
cause a gas nucleus to grow into a bubble?"*

In practice:
- VPM-B tends to produce **deeper first stops** (bubble nucleation argument)
- Bühlmann (with GF < 1) can produce **similar deep stops** but via a different mechanism
- For shallow recreational profiles, the two often agree
- For deep trimix dives, VPM-B's crushing effect becomes significant and diverges from Bühlmann

Neither model includes a scheduling algorithm in this tool — ceilings are shown as
instantaneous snapshots at the end of the bottom phase.
        """)

st.divider()
st.caption(
    "🫧 Decompression Teaching Tool v3 · ZHL-16C + VPM-B · "
    "Off-gassing depth and ceiling only — no deco scheduling · Educational use only"
)
