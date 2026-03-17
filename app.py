"""
Decompression Teaching Tool v4 by Carlos Lander
================================
Two calculations only:
  1. Off-gassing Start Depth  — VPM: depth where leading tissue first becomes
                                supersaturated (P_tissue > P_ambient)
  2. Bühlmann Ceiling Depth   — ZHL-16C: minimum ambient pressure at which no
                                tissue exceeds its M-value (GF = 1.0, pure model)

No deco scheduling. No stop times. No gradient factors.
"""

import math
import numpy as np
import pandas as pd
import streamlit as st

# =============================================================================
# ZHL-16C COEFFICIENTS  (Bühlmann 1995)
# =============================================================================
#
# 16 compartments. Each has:
#   - N2 and He half-times (minutes)
#   - a coefficient (bar)  — M-value y-intercept
#   - b coefficient (−)    — M-value slope reciprocal
#
# M-value line:  P_tol = a + P_amb / b
# Ceiling:       P_amb_min = (P_tissue − a) * b      [derived below]

N_COMP = 16

HT_N2 = np.array([
     4.0,   8.0,  12.5,  18.5,  27.0,  38.3,  54.3,  77.0,
   109.0, 146.0, 187.0, 239.0, 305.0, 390.0, 498.0, 635.0
])
HT_HE = np.array([
     1.51,   3.02,   4.72,   6.99,  10.21,  14.48,  20.53,  29.11,
    41.20,  55.19,  70.69,  90.34, 115.29, 147.42, 188.24, 240.03
])

# ZHL-16C a-values (bar)
A_N2 = np.array([
    1.2599, 1.0000, 0.8618, 0.7562, 0.6200, 0.5043, 0.4410, 0.4000,
    0.3750, 0.3500, 0.3295, 0.3065, 0.2835, 0.2610, 0.2480, 0.2327
])
A_HE = np.array([
    1.7424, 1.3830, 1.1919, 1.0458, 0.9220, 0.8205, 0.7305, 0.6502,
    0.5950, 0.5545, 0.5333, 0.5189, 0.5181, 0.5176, 0.5172, 0.5119
])

# ZHL-16C b-values (dimensionless)
B_N2 = np.array([
    0.5050, 0.6514, 0.7222, 0.7825, 0.8126, 0.8434, 0.8693, 0.8910,
    0.9092, 0.9222, 0.9319, 0.9403, 0.9477, 0.9544, 0.9602, 0.9653
])
B_HE = np.array([
    0.4245, 0.5747, 0.6527, 0.7223, 0.7582, 0.7957, 0.8279, 0.8553,
    0.8757, 0.8903, 0.8997, 0.9073, 0.9122, 0.9171, 0.9217, 0.9267
])

# Pre-computed rate constants  k = ln2 / half-time
K_N2 = np.log(2) / HT_N2
K_HE = np.log(2) / HT_HE

# =============================================================================
# ENVIRONMENT CONSTANTS
# =============================================================================
P_SURF     = 1.013    # bar  (surface pressure, sea level)
BAR_PER_M  = 0.09985  # bar/m seawater (1025 kg m⁻³ × g)
P_WVP      = 0.0627   # bar  alveolar water vapour at 37 °C

# VPM-B parameters (Yount & Hoffman 1986)
GAMMA_C   = 0.0179   # N/m  critical surface tension
R0_N2     = 1.0e-6   # m    initial bubble nucleus radius for N2 (1.0 µm)
R0_HE     = 1.0e-6   # m    initial bubble nucleus radius for He (1.0 µm)
LAMBDA_CE = 7500.0   # dimensionless skin-compression coefficient (N/m per bar)

# =============================================================================
# HELPER: depth ↔ pressure
# =============================================================================

def depth2p(d):
    """Absolute pressure (bar) at depth d (m)."""
    return P_SURF + d * BAR_PER_M

def p2depth(p):
    """Depth (m) from absolute pressure (bar).  Returns array or scalar."""
    return np.maximum(0.0, (p - P_SURF) / BAR_PER_M)

# =============================================================================
# SCHREINER GAS LOADING  (constant depth segment)
# =============================================================================

def schreiner(depth, t_min, fn2, fhe, n2_0, he_0):
    """
    Exact Haldanian loading for one constant-depth segment of t_min minutes.

    P_t = P_insp + (P_0 - P_insp) * exp(-k * t)

    Works for both on-gassing and off-gassing (direction is automatic).

    Returns updated (n2, he) tissue arrays.
    """
    p_amb = depth2p(depth)
    p_dry = max(0.0, p_amb - P_WVP)

    p_insp_n2 = fn2 * p_dry
    p_insp_he = fhe * p_dry

    n2 = p_insp_n2 + (n2_0 - p_insp_n2) * np.exp(-K_N2 * t_min)
    he = p_insp_he + (he_0 - p_insp_he) * np.exp(-K_HE * t_min)
    return n2, he

# =============================================================================
# BÜHLMANN M-VALUE CEILING  (GF = 1.0, pure model)
# =============================================================================
#
# M-value line (Bühlmann):
#     P_tol = a + P_amb / b
#
# Rearranged to find minimum ambient pressure for tissue tension P_t:
#     P_t ≤ a + P_amb / b
#     P_amb ≥ (P_t − a) * b
#
# So the minimum ambient pressure that keeps compartment i in tolerance:
#     P_min_i = (P_t_i − a_i) * b_i
#
# If P_min_i ≤ P_SURF the compartment is safe at the surface.
# The overall ceiling is the maximum P_min across all compartments,
# converted back to depth.
#
# For trimix, a and b are weighted by partial pressures (Baker method):
#     a_mix = (a_n2 * P_n2 + a_he * P_he) / (P_n2 + P_he)
#     b_mix = (b_n2 * P_n2 + b_he * P_he) / (P_n2 + P_he)

def mixed_ab(n2, he):
    """
    Partial-pressure-weighted a and b for each compartment.
    Falls back to pure N2 coefficients when He = 0.
    """
    pt = n2 + he
    # Avoid division by zero in pure O2 tissues (shouldn't happen, but safe)
    safe = pt > 1e-9
    a = np.where(safe, (A_N2 * n2 + A_HE * he) / np.where(safe, pt, 1.0), A_N2)
    b = np.where(safe, (B_N2 * n2 + B_HE * he) / np.where(safe, pt, 1.0), B_N2)
    return a, b


def buhlmann_ceiling(n2, he):
    """
    Bühlmann ZHL-16C ceiling (GF = 1.0).

    Returns
    -------
    ceiling_depth  : float, metres  (0 if all tissues safe at surface)
    leading_comp   : int, 0-based compartment index driving the ceiling
    p_min_per_comp : ndarray (16,), minimum ambient pressure per compartment (bar)
    """
    a, b = mixed_ab(n2, he)
    pt = n2 + he

    # Minimum tolerable ambient pressure per compartment (bar, absolute)
    # M-value line: P_tol = a + P_amb/b  →  P_amb_min = (P_tissue - a) * b
    p_min = (pt - a) * b                        # bar absolute
    p_min_clipped = np.maximum(p_min, 0.0)      # < 0 means safe at surface

    leading = int(np.argmax(p_min_clipped))
    # p_min is absolute pressure → depth = (p_min - P_SURF) / BAR_PER_M
    ceiling = float(p2depth(p_min_clipped[leading]))

    return ceiling, leading, p_min_clipped

# =============================================================================
# VPM-B  OFF-GASSING START DEPTH
# =============================================================================
#
# VPM defines the "decompression zone" as any depth where at least one tissue
# compartment is supersaturated — i.e. P_tissue > P_ambient.
#
# The off-gassing start depth is the DEEPEST depth at which this first occurs
# during a direct ascent from the bottom.  It equals the depth at which the
# most-loaded compartment crosses the ambient-pressure line.
#
# Per compartment:
#     P_tissue_i = P_amb  →  depth_i = (P_tissue_i − P_SURF) / BAR_PER_M
#
# This is the "decompression floor" used in VPM source code
# (Baker 1998 VPM Fortran: "START OF DECOMPRESSION ZONE").
#
# No bubble nucleation radius calculation is needed for this depth —
# that is only needed for computing the ceiling (which is Bühlmann here).

def vpm_offgas_depth(n2, he):
    """
    VPM decompression floor: depth where the most-loaded compartment first
    becomes supersaturated relative to ambient pressure.

    Returns
    -------
    offgas_depth  : float, metres
    leading_comp  : int, 0-based
    depth_per_comp: ndarray (16,), supersaturation-onset depth per compartment
    """
    pt = n2 + he
    # Depth at which each compartment exactly equals ambient
    depths = p2depth(pt)            # (P_t − P_SURF) / BAR_PER_M, clipped ≥ 0
    leading = int(np.argmax(depths))
    return float(depths[leading]), leading, depths

# =============================================================================
# DIVE SIMULATION  (descent + bottom only)
# =============================================================================

def simulate(max_depth, bottom_time, fo2, fn2, fhe, dt=0.25):
    """
    Simulates descent at DESCENT_RATE then bottom phase, returns final tissue state.

    Parameters
    ----------
    max_depth    : float, m
    bottom_time  : float, min
    fo2/fn2/fhe  : gas fractions (must sum to 1)
    dt           : time step, min

    Returns
    -------
    n2, he : ndarray (16,)  tissue inert gas pressures after bottom phase
    """
    DESCENT_RATE = 20.0  # m/min

    # Surface equilibrium: tissues saturated with air at surface
    #   P_n2_surface = 0.7902 * (P_SURF - P_WVP)
    p_dry_surf = P_SURF - P_WVP
    n2 = np.full(N_COMP, 0.7902 * p_dry_surf)
    he = np.zeros(N_COMP)

    # --- Descent (Schreiner at each step, depth increases linearly) ---
    descent_time  = max_depth / DESCENT_RATE
    n_steps       = max(1, int(descent_time / dt))
    for i in range(n_steps):
        frac  = (i + 0.5) / n_steps       # mid-step depth fraction
        depth = frac * max_depth
        n2, he = schreiner(depth, dt, fn2, fhe, n2, he)

    # Final half-step to land exactly at max_depth
    n2, he = schreiner(max_depth, descent_time - n_steps * dt + 1e-9, fn2, fhe, n2, he)

    # --- Bottom phase ---
    n_bottom = max(1, int(bottom_time / dt))
    for _ in range(n_bottom):
        n2, he = schreiner(max_depth, dt, fn2, fhe, n2, he)

    return n2, he

# =============================================================================
# STREAMLIT APP
# =============================================================================

st.set_page_config(
    page_title="Deco Teaching Tool",
    page_icon="🫧",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.card {
    background: #0e1c2f;
    border: 1px solid #1c3d60;
    border-radius: 10px;
    padding: 18px 22px;
    text-align: center;
}
.card-label { font-size:0.70rem; color:#5b8fb9; text-transform:uppercase;
              letter-spacing:0.10em; margin-bottom:6px; }
.card-value { font-size:2.0rem; font-weight:700; color:#e8f4ff; line-height:1.1; }
.card-sub   { font-size:0.72rem; color:#6a99bf; margin-top:5px; }
.card-warn  { border-color:#7a3010; }
.card-ok    { border-color:#1a7a3a; }

.formula-box {
    background:#091522;
    border-left: 3px solid #2a6eb5;
    border-radius:6px;
    padding:12px 16px;
    font-family: monospace;
    font-size:0.88rem;
    color:#a8d0f0;
    margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Dive Profile")

    fo2_pct = st.slider("Oxygen (%)", 5, 100, 21)
    fhe_pct = st.slider("Helium (%)",  0,  95,  0)
    fn2_pct = 100 - fo2_pct - fhe_pct

    if fn2_pct < 0:
        st.error("❌ O₂ + He > 100%")
        st.stop()

    fo2 = fo2_pct / 100.0
    fhe = fhe_pct / 100.0
    fn2 = fn2_pct / 100.0

    c1, c2, c3 = st.columns(3)
    c1.metric("O₂", f"{fo2_pct}%")
    c2.metric("He", f"{fhe_pct}%")
    c3.metric("N₂", f"{fn2_pct}%")

    st.divider()
    max_depth   = st.number_input("Max Depth (m)",     1.0, 330.0,  40.0, 1.0)
    bottom_time = st.number_input("Bottom Time (min)", 1.0, 480.0,  35.0, 1.0)

    st.divider()
    go = st.button("🔄 Calculate", use_container_width=True, type="primary")

# ── Run simulation ────────────────────────────────────────────────────────────
if go or "res" not in st.session_state:
    n2, he = simulate(max_depth, bottom_time, fo2, fn2, fhe)

    offgas_depth, offgas_comp, offgas_per_comp = vpm_offgas_depth(n2, he)
    ceil_depth,   ceil_comp,   ceil_per_comp   = buhlmann_ceiling(n2, he)

    st.session_state["res"] = dict(
        n2=n2, he=he,
        offgas_depth=offgas_depth, offgas_comp=offgas_comp,
        offgas_per_comp=offgas_per_comp,
        ceil_depth=ceil_depth, ceil_comp=ceil_comp,
        ceil_per_comp=ceil_per_comp,
        max_depth=max_depth, bottom_time=bottom_time,
        fo2=fo2, fn2=fn2, fhe=fhe,
    )

res = st.session_state["res"]

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("🫧 Decompression Teaching Tool")
st.caption(
    "**VPM** → off-gassing start depth  |  "
    "**Bühlmann ZHL-16C** → ceiling depth  |  "
    "No gradient factors  ·  No stop scheduling  ·  Educational use only  ·  by Carlos Lander"
)

st.divider()

# ── KPI Cards ─────────────────────────────────────────────────────────────────
k1, k2 = st.columns(2)

with k1:
    oc = res["offgas_comp"]
    practical_offgas = math.floor(res["offgas_depth"] / 3.0) * 3 if res["offgas_depth"] > 0 else 0
    st.markdown(f"""
    <div class="card">
      <div class="card-label">🔵 VPM — Off-gassing Start Depth</div>
      <div class="card-value">{res['offgas_depth']:.1f} m</div>
      <div class="card-sub">
        Nearest 3 m stop above: <strong>{practical_offgas:.0f} m</strong><br>
        Leading compartment: T{oc+1}
        &nbsp;(N₂ half-time {HT_N2[oc]:.0f} min)
      </div>
    </div>""", unsafe_allow_html=True)

with k2:
    cc = res["ceil_comp"]
    practical_ceil = math.ceil(res["ceil_depth"] / 3.0) * 3 if res["ceil_depth"] > 0 else 0
    warn = "card-warn" if res["ceil_depth"] > 0 else "card-ok"
    label = "⚠️ Decompression Required" if res["ceil_depth"] > 0 else "✅ No Decompression Required"
    st.markdown(f"""
    <div class="card {warn}">
      <div class="card-label">🔴 Bühlmann — Ceiling Depth</div>
      <div class="card-value">{res['ceil_depth']:.1f} m</div>
      <div class="card-sub">
        {label}<br>
        Nearest 3 m stop above: <strong>{practical_ceil:.0f} m</strong><br>
        Leading compartment: T{cc+1}
        &nbsp;(N₂ half-time {HT_N2[cc]:.0f} min)
      </div>
    </div>""", unsafe_allow_html=True)

st.divider()

# ── Zone diagram ──────────────────────────────────────────────────────────────
st.subheader("Depth Zones")

zone_df = pd.DataFrame({
    "Zone": [
        "Surface (0 m)",
        f"Bühlmann Ceiling ({res['ceil_depth']:.1f} m)",
        f"VPM Off-gassing Start ({res['offgas_depth']:.1f} m)",
        f"Bottom ({max_depth:.0f} m)",
    ],
    "Depth (m)": [0.0, res["ceil_depth"], res["offgas_depth"], max_depth],
})
st.bar_chart(zone_df, x="Zone", y="Depth (m)", horizontal=True, height=220)
st.caption(
    "**Between Off-gassing Start and Ceiling** is the safe ascent window — "
    "tissues are releasing gas but not yet violating the M-value limit. "
    "**Above the Ceiling** the leading tissue exceeds its Bühlmann M-value."
)

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🫁 All 16 Compartments",
    "📊 Ceiling per Compartment",
    "📖 Physics Reference",
])

# ─── Tab 1: full tissue table ─────────────────────────────────────────────────
with tab1:
    n2  = res["n2"]
    he  = res["he"]
    pt  = n2 + he
    a, b = mixed_ab(n2, he)

    # M-value at surface  =  a + P_SURF / b
    mv_surf = a + P_SURF / b

    # Saturation %  =  P_tissue / M_surface * 100
    sat = np.clip(pt / mv_surf * 100, 0, 999)

    # Is this compartment supersaturated at surface?
    super_at_surf = pt > P_SURF

    df = pd.DataFrame({
        "Tissue":                [f"T{i+1}" for i in range(N_COMP)],
        "N₂ ht (min)":           HT_N2.astype(int),
        "He ht (min)":           HT_HE,
        "N₂ load (bar)":         np.round(n2,  4),
        "He load (bar)":         np.round(he,  4),
        "Total P_t (bar)":       np.round(pt,  4),
        "M-val @ surface (bar)": np.round(mv_surf, 4),
        "Saturation %":          np.round(sat, 1),
        "Supersaturated?":       ["YES ⚠️" if s else "no" for s in super_at_surf],
    })
    # p_min = (Pt-a)*b is already absolute pressure (bar)
    # depth = (p_min - P_SURF) / BAR_PER_M  i.e. just p2depth(p_min)
    p_min = res["ceil_per_comp"]
    df["Bhl ceil depth (m)"] = np.round(p2depth(p_min), 2)

    st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Total Inert Gas vs Bühlmann M-value (at surface)")
    chart = pd.DataFrame({
        "Tissue":           [f"T{i+1}" for i in range(N_COMP)],
        "Total P_t (bar)":  np.round(pt, 4),
        "M-val @ surf (bar)": np.round(mv_surf, 4),
    })
    st.bar_chart(chart, x="Tissue", y=["Total P_t (bar)", "M-val @ surf (bar)"])
    st.caption(
        "Where Total P_t exceeds the M-value the compartment violates its surface limit — "
        "it needs ambient pressure (depth) to stay in tolerance."
    )

# ─── Tab 2: per-compartment ceiling depths ────────────────────────────────────
with tab2:
    p_min = res["ceil_per_comp"]
    comp_depths = p2depth(p_min)   # p_min is absolute pressure; p2depth = (p - P_SURF)/BAR_PER_M
    offgas_comp_depths = res["offgas_per_comp"]

    st.subheader("Bühlmann Ceiling Depth per Compartment")
    ceil_df = pd.DataFrame({
        "Tissue":              [f"T{i+1}" for i in range(N_COMP)],
        "Bhl Ceiling (m)":     np.round(comp_depths, 2),
        "VPM Offgas Start (m)": np.round(offgas_comp_depths, 2),
    })
    st.bar_chart(ceil_df, x="Tissue",
                 y=["Bhl Ceiling (m)", "VPM Offgas Start (m)"])
    st.caption(
        "Blue = depth at which each compartment exceeds its Bühlmann M-value (ceiling). "
        "Orange = depth at which each compartment first becomes supersaturated (VPM floor). "
        f"The overall ceiling is the maximum blue bar: **T{res['ceil_comp']+1} at {res['ceil_depth']:.1f} m**. "
        f"The overall off-gassing start is the maximum orange bar: **T{res['offgas_comp']+1} at {res['offgas_depth']:.1f} m**."
    )

    st.divider()
    st.subheader("Detail Table")
    detail = pd.DataFrame({
        "Tissue":               [f"T{i+1}" for i in range(N_COMP)],
        "N₂ ht (min)":          HT_N2.astype(int),
        "P_tissue (bar)":       np.round(res["n2"] + res["he"], 4),
        "Bhl Ceiling (m)":      np.round(comp_depths, 2),
        "VPM Offgas Start (m)": np.round(offgas_comp_depths, 2),
        "Window (m)":           np.round(
                                    np.maximum(0, offgas_comp_depths - comp_depths), 2
                                ),
    })
    st.dataframe(detail, use_container_width=True, hide_index=True)
    st.caption(
        "**Window** = Offgas Start − Ceiling per compartment. "
        "Positive window means there is a safe zone to off-gas before hitting the M-value."
    )

# ─── Tab 3: physics reference ─────────────────────────────────────────────────
with tab3:
    st.subheader("📖 Exact Formulas Used")

    st.markdown("#### 1. Gas Loading — Schreiner Equation (constant depth)")
    st.markdown("""
At each simulation step the tissue pressure is updated with:

$$P_t = P_{insp} + (P_{t,0} - P_{insp}) \\cdot e^{-k \\cdot t}$$

| Symbol | Value |
|--------|-------|
| $P_{insp} = f_{gas}\\,(P_{amb} - P_{H_2O})$ | Alveolar inspired partial pressure |
| $P_{H_2O}$ | 0.0627 bar (water vapour, 37 °C) |
| $k = \\ln 2 \\,/\\, t_{\\frac{1}{2}}$ | Rate constant from half-time |
| $t$ | Exposure time (min) |

Applied for descent (ramping depth) and for the full bottom phase.
""")

    st.markdown('<div class="formula-box">'
        'n2[i] = p_insp_n2 + (n2_0[i] - p_insp_n2) * exp(-k_n2[i] * t)<br>'
        'he[i] = p_insp_he + (he_0[i] - p_insp_he) * exp(-k_he[i] * t)'
        '</div>', unsafe_allow_html=True)

    st.divider()

    st.markdown("#### 2. VPM — Off-gassing Start Depth")
    st.markdown("""
The VPM decompression zone begins where tissue pressure first exceeds ambient:

$$P_{tissue,i} > P_{ambient}$$

Per compartment, the off-gassing onset depth is:

$$d_{i} = \\frac{P_{tissue,i} - P_{surface}}{\\rho g} = \\frac{P_{tissue,i} - 1.013}{0.09985}$$

The **overall off-gassing start depth** is the maximum $d_i$ across all 16 compartments.
This is the depth at which the first (most-loaded) compartment crosses the ambient
pressure line — the decompression floor as defined in Baker's VPM Fortran source.

> No bubble radius calculation is required for this depth.
> The radius/nucleus tracking in VPM is used when computing the *ceiling*, which here
> is handled by Bühlmann.
""")

    st.markdown('<div class="formula-box">'
        'pt[i]   = n2[i] + he[i]<br>'
        'depth_i = max(0,  (pt[i] - P_SURF) / BAR_PER_M)<br>'
        'offgas_depth = max(depth_i for i in 1..16)'
        '</div>', unsafe_allow_html=True)

    st.divider()

    st.markdown("#### 3. Bühlmann ZHL-16C — Ceiling Depth  (GF = 1.0)")
    st.markdown("""
The Bühlmann M-value line for compartment $i$:

$$P_{tol,i} = a_i + \\frac{P_{amb}}{b_i}$$

Rearranged — the minimum ambient pressure that keeps $P_t \\leq P_{tol}$:

$$P_{amb,min,i} = (P_{t,i} - a_i) \\cdot b_i$$

For **Trimix** the coefficients are weighted by partial pressure
(Baker / Bühlmann method):

$$a_{mix} = \\frac{a_{N_2}\\,P_{N_2} + a_{He}\\,P_{He}}{P_{N_2}+P_{He}}
\\qquad
b_{mix} = \\frac{b_{N_2}\\,P_{N_2} + b_{He}\\,P_{He}}{P_{N_2}+P_{He}}$$

The overall ceiling depth:

$$d_{ceil} = \\max_i\\left[\\frac{P_{amb,min,i} - P_{surface}}{0.09985}\\right]$$
""")

    st.markdown('<div class="formula-box">'
        'a, b    = mixed_ab(n2, he)          # weighted by partial pressure<br>'
        'p_min_i = max(0,  (n2[i]+he[i] - a[i]) * b[i])<br>'
        'ceiling = max(p2depth(p_min_i + P_SURF) for i in 1..16)'
        '</div>', unsafe_allow_html=True)

    st.divider()

    st.markdown("#### Key distinction between the two depths")
    st.markdown("""
| | Off-gassing Start (VPM floor) | Ceiling (Bühlmann) |
|---|---|---|
| **Condition** | $P_t > P_{amb}$ | $P_t > a + P_{amb}/b$ |
| **Meaning** | Gas begins leaving tissue | M-value exceeded — risk of DCS |
| **Model** | VPM (decompression zone definition) | Bühlmann ZHL-16C |
| **Between them** | Safe off-gassing window | — |
| **Above ceiling** | ⚠️ Dangerous | — |

The ceiling is always **shallower** than (or equal to) the off-gassing start depth,
because the M-value line sits above the ambient pressure line.
""")

st.divider()
st.caption(
    "🫧 Decompression Teaching Tool v4  ·  "
    "VPM off-gassing depth + Bühlmann ZHL-16C ceiling  ·  "
    "GF = 1.0 (pure model)  ·  No deco scheduling  ·  Educational use only"
)
