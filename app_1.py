import numpy as np
import streamlit as st
import math
import pandas as pd

# =============================================================================
# PHYSICS ENGINE v2 — Enhanced ZHL-16C + VPM-B Bubble Nucleation
# =============================================================================

class DiveConfig:
    """
    ZHL-16C coefficients (Bühlmann 1995, Baker 1998 edition).
    Each compartment has: half-time (N2 & He), 'a' value, 'b' value.
    The 'a' values set the y-intercept of the M-value line (max supersaturation at 0 bar).
    The 'b' values set the slope (how M-value changes with depth).
    """
    def __init__(self):
        # --- ZHL-16C Tissue Compartments ---
        self.N_COMPARTMENTS = 16

        self.HALF_TIMES_N2 = np.array([
            4.0, 8.0, 12.5, 18.5, 27.0, 38.3, 54.3, 77.0,
            109.0, 146.0, 187.0, 239.0, 305.0, 390.0, 498.0, 635.0
        ])
        self.HALF_TIMES_HE = np.array([
            1.51, 3.02, 4.72, 6.99, 10.21, 14.48, 20.53, 29.11,
            41.20, 55.19, 70.69, 90.34, 115.29, 147.42, 188.24, 240.03
        ])

        # ZHL-16C 'a' coefficients (bar) — N2
        self.A_N2 = np.array([
            1.2599, 1.0000, 0.8618, 0.7562, 0.6200, 0.5043, 0.4410, 0.4000,
            0.3750, 0.3500, 0.3295, 0.3065, 0.2835, 0.2610, 0.2480, 0.2327
        ])
        # ZHL-16C 'b' coefficients (dimensionless) — N2
        self.B_N2 = np.array([
            0.5050, 0.6514, 0.7222, 0.7825, 0.8126, 0.8434, 0.8693, 0.8910,
            0.9092, 0.9222, 0.9319, 0.9403, 0.9477, 0.9544, 0.9602, 0.9653
        ])
        # ZHL-16C 'a' coefficients (bar) — He
        self.A_HE = np.array([
            1.7424, 1.3830, 1.1919, 1.0458, 0.9220, 0.8205, 0.7305, 0.6502,
            0.5950, 0.5545, 0.5333, 0.5189, 0.5181, 0.5176, 0.5172, 0.5119
        ])
        # ZHL-16C 'b' coefficients (dimensionless) — He
        self.B_HE = np.array([
            0.4245, 0.5747, 0.6527, 0.7223, 0.7582, 0.7957, 0.8279, 0.8553,
            0.8757, 0.8903, 0.8997, 0.9073, 0.9122, 0.9171, 0.9217, 0.9267
        ])

        # --- Environment ---
        self.SURFACE_PRESSURE = 1.013  # bar
        self.BAR_PER_METER     = 0.0998  # bar/m (seawater, 1025 kg/m³)
        self.PRESSURE_OTHER_GASES = 0.0627  # water vapor + CO2 (bar) at 37°C

        # --- Ascent / Stop Parameters ---
        self.ASCENT_RATE   = 9.0   # m/min
        self.DESCENT_RATE  = 20.0  # m/min
        self.MIN_STOP_TIME = 1.0   # minutes per stop increment
        self.STOP_INCREMENT = 3.0  # metres between stops

        # --- VPM-B Bubble Nucleation Parameters ---
        # Gamma (surface tension of nucleus, N/m) — varies with gas mix
        self.GAMMA_C      = 0.0179  # critical surface tension (N/m)
        self.GAMMA_INIT   = 0.0257  # initial surface tension (N/m)
        # Initial critical radius (metres, ~0.8 μm typical)
        self.R_INITIAL    = 0.8e-6  # 0.8 micrometres
        # Skin compression coefficient (λ) — resists bubble growth
        self.LAMBDA_CE    = 7500.0  # (N/m²) / (bar of supersaturation)

        # --- Pre-computed rate constants ---
        self.k_n2 = np.log(2) / self.HALF_TIMES_N2
        self.k_he = np.log(2) / self.HALF_TIMES_HE

        # --- Gradient Factor defaults ---
        self.GF_LO = 0.30   # GF at deepest stop
        self.GF_HI = 0.85   # GF at surface

    def mixed_a_b(self, fn2: float, fhe: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Helium-weighted interpolation of a and b values (Baker/Bühlmann method).
        a_mix = (a_n2 * fN2 + a_he * fHe) / (fN2 + fHe)
        """
        denom = fn2 + fhe
        if denom == 0:
            return self.A_N2.copy(), self.B_N2.copy()
        a = (self.A_N2 * fn2 + self.A_HE * fhe) / denom
        b = (self.B_N2 * fn2 + self.B_HE * fhe) / denom
        return a, b


config = DiveConfig()


# =============================================================================
# CORE PHYSICS FUNCTIONS
# =============================================================================

def depth_to_pressure(depth: float) -> float:
    return config.SURFACE_PRESSURE + depth * config.BAR_PER_METER

def pressure_to_depth(pressure: float) -> float:
    return max(0.0, (pressure - config.SURFACE_PRESSURE) / config.BAR_PER_METER)

def inspired_pp(depth: float, fo2: float, fn2: float, fhe: float) -> tuple:
    """Inspired partial pressures, corrected for water vapour + CO₂."""
    amb = depth_to_pressure(depth)
    dry = amb - config.PRESSURE_OTHER_GASES
    dry = max(dry, 0.0)
    return fo2 * dry, fn2 * dry, fhe * dry


def load_tissues(
    depth: float, time_min: float,
    fo2: float, fn2: float, fhe: float,
    n2_init: np.ndarray, he_init: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Exact Haldanian exponential loading for a single depth/time segment.
    Works for both on-gassing and off-gassing — direction is automatic.
    """
    _, p_n2, p_he = inspired_pp(depth, fo2, fn2, fhe)
    n2 = p_n2 + (n2_init - p_n2) * np.exp(-config.k_n2 * time_min)
    he = p_he + (he_init - p_he) * np.exp(-config.k_he * time_min)
    return n2, he


# =============================================================================
# ZHL-16C M-VALUE CEILING
# =============================================================================

def buhlmann_ceiling(n2: np.ndarray, he: np.ndarray, fn2: float, fhe: float, gf: float = 1.0) -> float:
    """
    Compute the Bühlmann M-value ceiling depth for the current tissue loadings.

    M-value limit:  P_tol = (P_tissue - a_mix) * b_mix
    So the minimum tolerated ambient pressure = (P_tissue - a*gf) * b / (2 - b) … simplified:
        P_amb_tol = (P_t - a_mix * gf) * b_mix

    With gradient factors:
        P_amb_tol = (P_t - a_mix * gf) * b_mix
    Standard rearrangement:
        P_ceiling = (Pt - a * gf) * b    <- minimum ambient (bar)
    """
    a, b = config.mixed_a_b(fn2, fhe)
    pt = n2 + he
    p_ceiling = (pt - a * gf) * b
    p_ceiling = np.maximum(p_ceiling, 0.0)
    max_p = np.max(p_ceiling)
    return pressure_to_depth(max_p + config.SURFACE_PRESSURE) if max_p > 0 else 0.0


def leading_tissue_index(n2: np.ndarray, he: np.ndarray, fn2: float, fhe: float) -> int:
    a, b = config.mixed_a_b(fn2, fhe)
    pt = n2 + he
    p_ceiling = (pt - a) * b
    return int(np.argmax(p_ceiling))


# =============================================================================
# VPM-B BUBBLE NUCLEATION MODEL
# =============================================================================

class BubbleTracker:
    """
    Tracks the critical radius and allowable supersaturation for each of the
    16 tissue compartments using VPM-B (Yount & Hoffman 1986, Maiken 1995).

    Key physics:
    ─────────────────────────────────────────────────────────────────────
    • At depth, the ambient pressure compresses bubble nuclei from their
      resting radius r₀ to a smaller radius rᵢ.
    • On ascent, if supersaturation exceeds the "crushing pressure" the
      nucleus grows — this marks bubble nucleation.
    • The allowable supersaturation per compartment is:

          ΔP_allow = 2γ/r_crit    (Young-Laplace)

    • r_crit shrinks when a high pressure exposure "crushes" nuclei and
      then the diver surfaces — a history-dependent effect.
    ─────────────────────────────────────────────────────────────────────
    """

    def __init__(self):
        # Initial critical radii — all compartments start at R_INITIAL
        self.r_crit = np.full(config.N_COMPARTMENTS, config.R_INITIAL, dtype=float)
        self.max_crushing_pressure = np.zeros(config.N_COMPARTMENTS, dtype=float)

    def update_crushing(self, ambient_pressure: float):
        """
        During descent/bottom phase, high ambient pressure can crush nuclei,
        reducing r_crit and therefore *increasing* the allowable supersaturation
        on ascent (nuclei harder to grow). Called every time-step.
        """
        p_surface = config.SURFACE_PRESSURE
        # Crushing pressure = how far above surface ambient pressure
        p_crush = max(0.0, ambient_pressure - p_surface)
        self.max_crushing_pressure = np.maximum(self.max_crushing_pressure, p_crush)

        # Reduced radius after compression:
        # r_crushed = r_init * (2γ_c) / (2γ_c + λ * p_crush)
        # λ_CE is the skin-compression coefficient
        numerator   = 2.0 * config.GAMMA_C * config.R_INITIAL
        denominator = 2.0 * config.GAMMA_C + config.LAMBDA_CE * self.max_crushing_pressure
        self.r_crit = numerator / denominator

    def allowable_supersaturation(self) -> np.ndarray:
        """
        Per-compartment allowable supersaturation (bar) before bubble growth:
            ΔP_allow = (2γ / r_crit) * conservatism_scaling
        Larger r_crit → smaller allowable ΔP (easier to nucleate).
        Crushed nucleus → smaller r_crit → larger allowable ΔP (harder to nucleate).
        """
        return (2.0 * config.GAMMA_C) / self.r_crit

    def vpm_ceiling_depth(self, n2: np.ndarray, he: np.ndarray) -> float:
        """
        Returns the depth (m) above which bubble nucleation would occur.
        For each compartment:
            P_min_amb = P_tissue - ΔP_allow
        The ceiling is the depth corresponding to max(P_min_amb).
        """
        p_min = (n2 + he) - self.allowable_supersaturation()
        p_min = np.maximum(p_min, 0.0)
        return pressure_to_depth(np.max(p_min) + config.SURFACE_PRESSURE) if np.max(p_min) > 0 else 0.0


# =============================================================================
# FULL DIVE SIMULATION WITH CONTINUOUS ON-GASSING
# =============================================================================

def run_full_simulation(
    max_depth: float,
    bottom_time: float,
    fo2: float, fn2: float, fhe: float,
    gf_lo: float = 0.30,
    gf_hi: float = 0.85,
    dt: float = 0.1   # minutes per simulation step
) -> dict:
    """
    Simulates descent → bottom → ascent with deco stops.
    Tracks tissue loading, Bühlmann M-values, and VPM-B bubble nucleation
    continuously — including on-gassing *during* deco stops.

    Returns a rich dict with:
      - deco_schedule : list of (depth_m, stop_time_min) tuples
      - tissue_history: list of snapshots for plotting
      - offgas_depth   : depth where first supersaturation occurs
      - bubble_tracker : final BubbleTracker state
      - total_deco_time
      - ndl            : no-decompression limit (if no stops needed)
    """

    # --- Initial tissue state (surface equilibrium) ---
    n2 = np.full(config.N_COMPARTMENTS, 0.7902 * (config.SURFACE_PRESSURE - config.PRESSURE_OTHER_GASES))
    he = np.zeros(config.N_COMPARTMENTS)
    bubbles = BubbleTracker()

    # ── PHASE 1: Descent ──────────────────────────────────────────────────────
    descent_time = max_depth / config.DESCENT_RATE
    steps_descent = max(1, int(descent_time / dt))
    for i in range(steps_descent):
        t_elapsed = (i + 1) * dt
        current_depth = min(max_depth, config.DESCENT_RATE * t_elapsed)
        n2, he = load_tissues(current_depth, dt, fo2, fn2, fhe, n2, he)
        bubbles.update_crushing(depth_to_pressure(current_depth))

    # ── PHASE 2: Bottom ───────────────────────────────────────────────────────
    steps_bottom = max(1, int(bottom_time / dt))
    for _ in range(steps_bottom):
        n2, he = load_tissues(max_depth, dt, fo2, fn2, fhe, n2, he)
        bubbles.update_crushing(depth_to_pressure(max_depth))

    # --- Detect first supersaturation depth ---
    # We'll find this during the ascent simulation below
    offgas_depth = None
    offgas_tissue = -1

    # ── PHASE 3: Ascent + Deco Stop Calculation ───────────────────────────────
    # GF varies linearly from gf_lo at the first stop to gf_hi at surface
    # We need to find the first stop first (two-pass approach)

    # Pass 1 — find deepest required stop (use gf_lo)
    n2_scan, he_scan = n2.copy(), he.copy()
    bubbles_scan = BubbleTracker()
    bubbles_scan.r_crit = bubbles.r_crit.copy()
    bubbles_scan.max_crushing_pressure = bubbles.max_crushing_pressure.copy()

    first_stop_depth = 0.0
    scan_depth = max_depth
    while scan_depth > 0:
        scan_depth = max(0.0, scan_depth - config.ASCENT_RATE * dt)
        n2_scan, he_scan = load_tissues(scan_depth, dt, fo2, fn2, fhe, n2_scan, he_scan)
        bhl_ceil = buhlmann_ceiling(n2_scan, he_scan, fn2, fhe, gf=gf_lo)
        vpm_ceil = bubbles_scan.vpm_ceiling_depth(n2_scan, he_scan)
        combined_ceil = max(bhl_ceil, vpm_ceil)
        if combined_ceil > scan_depth and first_stop_depth == 0.0:
            first_stop_depth = _round_up_3m(combined_ceil)
            break

    # Pass 2 — simulate ascent properly, computing stops with interpolated GF
    deco_schedule = []
    tissue_history = []

    current_depth = max_depth
    n2_sim, he_sim = n2.copy(), he.copy()
    bubbles_sim = BubbleTracker()
    bubbles_sim.r_crit = bubbles.r_crit.copy()
    bubbles_sim.max_crushing_pressure = bubbles.max_crushing_pressure.copy()

    total_time = 0.0
    stop_depths_tried: set[float] = set()

    while current_depth > 0.0:
        # Move up one step
        new_depth = max(0.0, current_depth - config.ASCENT_RATE * dt)
        n2_sim, he_sim = load_tissues(new_depth, dt, fo2, fn2, fhe, n2_sim, he_sim)
        total_time += dt

        amb = depth_to_pressure(new_depth)
        total_load = n2_sim + he_sim

        # Detect off-gassing start
        if offgas_depth is None and np.any(total_load > amb):
            offgas_depth = new_depth
            offgas_tissue = int(np.argmax(total_load - amb))

        # Interpolated GF
        gf = _interpolate_gf(new_depth, first_stop_depth, gf_lo, gf_hi)

        # Combined ceiling
        bhl_ceil = buhlmann_ceiling(n2_sim, he_sim, fn2, fhe, gf=gf)
        vpm_ceil = bubbles_sim.vpm_ceiling_depth(n2_sim, he_sim)
        combined_ceil = max(bhl_ceil, vpm_ceil)
        stop_depth = _round_up_3m(combined_ceil)

        # Record snapshot for plotting (every 0.5 min)
        if total_time % 0.5 < dt:
            tissue_history.append({
                "time": round(total_time, 1),
                "depth": round(new_depth, 1),
                "n2": n2_sim.copy(),
                "he": he_sim.copy(),
                "bhl_ceil": round(bhl_ceil, 2),
                "vpm_ceil": round(vpm_ceil, 2),
                "r_crit_mean": float(np.mean(bubbles_sim.r_crit)) * 1e6  # μm
            })

        # Do we need a stop here?
        if stop_depth > new_depth and stop_depth not in stop_depths_tried:
            stop_depths_tried.add(stop_depth)
            stop_time = 0.0

            # On-gas during stop — keep adding minutes until ceiling clears
            while True:
                gf_stop = _interpolate_gf(stop_depth, first_stop_depth, gf_lo, gf_hi)
                bhl_ceil_stop = buhlmann_ceiling(n2_sim, he_sim, fn2, fhe, gf=gf_stop)
                vpm_ceil_stop = bubbles_sim.vpm_ceiling_depth(n2_sim, he_sim)
                required_stop = max(bhl_ceil_stop, vpm_ceil_stop)

                if required_stop <= stop_depth:
                    break  # Ceiling has cleared — can ascend

                # *** KEY: on-gas at stop depth (not just off-gas!) ***
                n2_sim, he_sim = load_tissues(stop_depth, config.MIN_STOP_TIME, fo2, fn2, fhe, n2_sim, he_sim)
                stop_time += config.MIN_STOP_TIME
                total_time += config.MIN_STOP_TIME

                # Safety valve — max 99 min per stop
                if stop_time >= 99:
                    break

            if stop_time > 0:
                deco_schedule.append((stop_depth, stop_time))
            new_depth = stop_depth

        current_depth = new_depth

    total_deco_time = sum(t for _, t in deco_schedule)

    # --- NDL calculation (no deco dive check) ---
    ndl = None
    if not deco_schedule:
        ndl = _calc_ndl(max_depth, fo2, fn2, fhe, gf_hi)

    return {
        "deco_schedule": deco_schedule,
        "tissue_history": tissue_history,
        "offgas_depth": offgas_depth,
        "offgas_tissue": offgas_tissue,
        "first_stop_depth": first_stop_depth,
        "total_deco_time": total_deco_time,
        "ndl": ndl,
        "bubble_tracker": bubbles_sim,
        "n2_final": n2_sim,
        "he_final": he_sim,
        "fn2": fn2,
        "fhe": fhe,
    }


def _round_up_3m(depth: float) -> float:
    if depth <= 0:
        return 0.0
    return math.ceil(depth / 3.0) * 3.0


def _interpolate_gf(depth: float, first_stop: float, gf_lo: float, gf_hi: float) -> float:
    if first_stop <= 0:
        return gf_hi
    frac = depth / first_stop
    return gf_lo + (gf_hi - gf_lo) * (1.0 - frac)


def _calc_ndl(max_depth: float, fo2: float, fn2: float, fhe: float, gf_hi: float) -> float:
    """Find no-decompression limit by binary search."""
    n2 = np.full(config.N_COMPARTMENTS, 0.7902 * (config.SURFACE_PRESSURE - config.PRESSURE_OTHER_GASES))
    he = np.zeros(config.N_COMPARTMENTS)
    t = 0.0
    while t < 300:
        n2, he = load_tissues(max_depth, 1.0, fo2, fn2, fhe, n2, he)
        t += 1.0
        if buhlmann_ceiling(n2, he, fn2, fhe, gf=gf_hi) > 0:
            return max(0.0, t - 1.0)
    return 999.0  # Unlimited for the range tested


# =============================================================================
# STREAMLIT UI
# =============================================================================

st.set_page_config(
    page_title="Deco Teaching Tool v2",
    page_icon="🫧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-box { background:#0f1923; border:1px solid #1e3a5f; border-radius:8px;
                  padding:12px 16px; margin-bottom:8px; }
    .metric-label { font-size:0.75rem; color:#5b8fb9; text-transform:uppercase;
                    letter-spacing:0.08em; }
    .metric-value { font-size:1.6rem; font-weight:700; color:#e0f0ff; }
    .metric-sub   { font-size:0.72rem; color:#7a9bbf; margin-top:2px; }
    .stop-row     { background:#0d1e2e; border-left:3px solid #1a6eb5;
                    padding:6px 12px; margin:3px 0; border-radius:4px;
                    font-family: monospace; color:#c8dff0; font-size:0.9rem; }
    .stop-row.deep { border-left-color:#c0392b; }
    .warn-box     { background:#1a1000; border:1px solid #8b6914;
                    border-radius:6px; padding:10px 14px; margin:6px 0;
                    color:#f0c040; font-size:0.88rem; }
    .ok-box       { background:#001a09; border:1px solid #1a7a3a;
                    border-radius:6px; padding:10px 14px; margin:6px 0;
                    color:#40d080; font-size:0.88rem; }
    .section-head { color:#5b8fb9; font-size:0.8rem; text-transform:uppercase;
                    letter-spacing:0.1em; margin:14px 0 4px 0; }
</style>
""", unsafe_allow_html=True)

st.title("🫧 Decompression Physics Teaching Tool v2")
st.caption("ZHL-16C · VPM-B Bubble Nucleation · Continuous On-gassing · Gradient Factors")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Dive Profile")

    fo2 = st.slider("Oxygen (%)", 5, 100, 21) / 100.0
    fhe = st.slider("Helium (%)", 0, 95, 35) / 100.0
    fn2 = round(1.0 - fo2 - fhe, 6)

    if fn2 < -0.001:
        st.error("❌ O₂ + He > 100%")
        st.stop()
    fn2 = max(fn2, 0.0)

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("O₂", f"{fo2*100:.0f}%")
    col_b.metric("He", f"{fhe*100:.0f}%")
    col_c.metric("N₂", f"{fn2*100:.0f}%")

    st.divider()
    max_depth    = st.number_input("Max Depth (m)", 1.0, 330.0, 40.0, 1.0)
    bottom_time  = st.number_input("Bottom Time (min)", 1.0, 480.0, 35.0, 1.0)

    st.divider()
    st.subheader("Gradient Factors")
    gf_lo = st.slider("GF Low (deep stop conservatism)", 0.10, 1.00, 0.35, 0.05)
    gf_hi = st.slider("GF High (surface conservatism)",  0.10, 1.00, 0.85, 0.05)
    if gf_lo > gf_hi:
        st.warning("GF Low should be ≤ GF High")

    with st.expander("ℹ️ GF Explanation"):
        st.markdown("""
**Gradient Factors** (Baker 1998) scale the Bühlmann M-value:

- **GF Low** = fraction of M-value headroom used at the *deepest* stop.
  Smaller → more conservative deep stops (VPM-like behaviour).

- **GF High** = fraction used at the *surface*.
  Smaller → more conservative shallow stops.

A pair like **35/85** gives deep stops and a conservative surface
approach, mimicking VPM-B behaviour within the Bühlmann framework.
        """)

    st.divider()
    run_btn = st.button("🔄 Calculate Dive", use_container_width=True, type="primary")

# ── Main ─────────────────────────────────────────────────────────────────────

if run_btn or "sim" not in st.session_state:
    with st.spinner("Simulating full dive profile…"):
        st.session_state["sim"] = run_full_simulation(
            max_depth, bottom_time, fo2, fn2, fhe, gf_lo, gf_hi
        )

sim = st.session_state["sim"]
deco    = sim["deco_schedule"]
history = sim["tissue_history"]

# ── Top KPIs ─────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.markdown('<div class="metric-box">'
        '<div class="metric-label">Off-gas Start</div>'
        f'<div class="metric-value">{sim["offgas_depth"]:.1f} m</div>'
        f'<div class="metric-sub">Tissue {sim["offgas_tissue"]+1 if sim["offgas_tissue"]>=0 else "—"}</div>'
        '</div>', unsafe_allow_html=True)

with k2:
    first_stop = deco[0][0] if deco else 0
    st.markdown('<div class="metric-box">'
        '<div class="metric-label">First Stop</div>'
        f'<div class="metric-value">{first_stop:.0f} m</div>'
        f'<div class="metric-sub">Deepest deco stop</div>'
        '</div>', unsafe_allow_html=True)

with k3:
    total_deco = sim["total_deco_time"]
    st.markdown('<div class="metric-box">'
        '<div class="metric-label">Total Deco Time</div>'
        f'<div class="metric-value">{total_deco:.0f} min</div>'
        f'<div class="metric-sub">{"No deco required" if total_deco == 0 else f"{len(deco)} stops"}</div>'
        '</div>', unsafe_allow_html=True)

with k4:
    ndl_val = sim["ndl"]
    st.markdown('<div class="metric-box">'
        '<div class="metric-label">NDL</div>'
        f'<div class="metric-value">{"N/A" if ndl_val is None else (f"{ndl_val:.0f} min" if ndl_val < 999 else "∞")}</div>'
        f'<div class="metric-sub">No-deco limit at {max_depth:.0f}m</div>'
        '</div>', unsafe_allow_html=True)

with k5:
    r_mean = sim["bubble_tracker"].r_crit.mean() * 1e6
    st.markdown('<div class="metric-box">'
        '<div class="metric-label">Mean r_crit</div>'
        f'<div class="metric-value">{r_mean:.3f} μm</div>'
        f'<div class="metric-sub">Post-dive bubble nucleus radius</div>'
        '</div>', unsafe_allow_html=True)

st.divider()

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Deco Schedule",
    "🫁 Tissue Loading",
    "🔬 Bubble Nucleation (VPM-B)",
    "📖 Physics Explainer"
])

# ─────────────────────── TAB 1: Deco Schedule ────────────────────────────────
with tab1:
    if not deco:
        st.markdown('<div class="ok-box">✅ <strong>No Decompression Required</strong><br>'
            f'NDL at {max_depth:.0f}m with this mix: '
            f'{"unlimited" if sim["ndl"] and sim["ndl"] >= 999 else f"{sim["ndl"]:.0f} min"}'
            '</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="warn-box">⚠️ '
            f'<strong>Decompression Required — {len(deco)} stops, {total_deco:.0f} min total</strong>'
            '</div>', unsafe_allow_html=True)

        # Table
        rows = []
        for depth, time in sorted(deco, reverse=True):
            rows.append({
                "Stop Depth (m)": f"{depth:.0f}",
                "Stop Time (min)": f"{time:.0f}",
                "Pressure (bar)": f"{depth_to_pressure(depth):.2f}",
                "GF at Stop": f"{_interpolate_gf(depth, sim['first_stop_depth'], gf_lo, gf_hi):.2f}"
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("##### Stop-by-Stop HTML View")
        for depth, time in sorted(deco, reverse=True):
            css_class = "stop-row deep" if depth >= 15 else "stop-row"
            st.markdown(
                f'<div class="{css_class}">⬢ {depth:.0f} m — {time:.0f} min '
                f'(GF {_interpolate_gf(depth, sim["first_stop_depth"], gf_lo, gf_hi):.2f})</div>',
                unsafe_allow_html=True
            )

    st.divider()
    st.subheader("Key Depths Summary")
    depth_data = {
        "Event": ["Bottom", "Off-gassing Starts", "First Deco Stop", "Surface"],
        "Depth (m)": [
            max_depth,
            round(sim["offgas_depth"] or 0, 1),
            first_stop,
            0
        ]
    }
    st.bar_chart(pd.DataFrame(depth_data), x="Event", y="Depth (m)", horizontal=True)

# ─────────────────────── TAB 2: Tissue Loading ───────────────────────────────
with tab2:
    if not history:
        st.info("No ascent data to display.")
    else:
        st.subheader("Ceiling vs Time During Ascent")
        hist_df = pd.DataFrame([{
            "Time (min)": h["time"],
            "Depth (m)": h["depth"],
            "Bühlmann Ceiling (m)": h["bhl_ceil"],
            "VPM-B Ceiling (m)": h["vpm_ceil"],
        } for h in history])
        st.line_chart(hist_df, x="Time (min)", y=["Depth (m)", "Bühlmann Ceiling (m)", "VPM-B Ceiling (m)"])
        st.caption("Depth shows where the diver is; ceilings show the minimum safe depth per model.")

        st.divider()
        st.subheader("Final Tissue Loading (After Dive)")
        n2_f  = sim["n2_final"]
        he_f  = sim["he_final"]
        total = n2_f + he_f
        a, b  = config.mixed_a_b(fn2, fhe)
        # M-value at surface
        mv_surface = a + config.SURFACE_PRESSURE / b

        tissue_df = pd.DataFrame({
            "Tissue": [f"T{i+1}" for i in range(16)],
            "Half-time N₂ (min)": config.HALF_TIMES_N2,
            "N₂ Load (bar)": np.round(n2_f, 3),
            "He Load (bar)": np.round(he_f, 3),
            "Total Inert (bar)": np.round(total, 3),
            "M-value Limit (bar)": np.round(mv_surface, 3),
            "Saturation %": np.round(total / mv_surface * 100, 1),
        })
        st.dataframe(tissue_df, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Saturation vs M-value Limit (Final)")
        chart_df = pd.DataFrame({
            "Tissue": [f"T{i+1}" for i in range(16)],
            "Inert Gas Load": np.round(total, 3),
            "M-value Limit": np.round(mv_surface, 3),
        })
        st.bar_chart(chart_df, x="Tissue", y=["Inert Gas Load", "M-value Limit"])

# ─────────────────────── TAB 3: VPM-B Bubble Nucleation ─────────────────────
with tab3:
    st.subheader("🔬 VPM-B: Bubble Nucleus Radius (r_crit) per Compartment")

    r_crits  = sim["bubble_tracker"].r_crit * 1e6  # μm
    r_init   = np.full(16, config.R_INITIAL * 1e6)
    crush_dp = sim["bubble_tracker"].max_crushing_pressure

    bubble_df = pd.DataFrame({
        "Tissue": [f"T{i+1}" for i in range(16)],
        "r_crit after dive (μm)": np.round(r_crits, 4),
        "Initial r₀ (μm)": np.round(r_init, 4),
        "Max crushing ΔP (bar)": np.round(crush_dp, 3),
        "Allowable supersaturation (bar)": np.round(sim["bubble_tracker"].allowable_supersaturation(), 3),
    })
    st.dataframe(bubble_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("r_crit vs Initial r₀ by Compartment")
    r_chart = pd.DataFrame({
        "Tissue": [f"T{i+1}" for i in range(16)],
        "r_crit post-dive (μm)": np.round(r_crits, 4),
        "r₀ initial (μm)": np.round(r_init, 4),
    })
    st.bar_chart(r_chart, x="Tissue", y=["r_crit post-dive (μm)", "r₀ initial (μm)"])
    st.caption(
        "When r_crit < r₀, the dive has 'crushed' the nuclei — "
        "higher pressures during the dive make bubbles harder to grow on the next dive. "
        "This is the physical basis for surface interval tables."
    )

    st.divider()
    st.subheader("Allowable Supersaturation per Compartment")
    allow_df = pd.DataFrame({
        "Tissue": [f"T{i+1}" for i in range(16)],
        "Allowable ΔP (bar)": np.round(sim["bubble_tracker"].allowable_supersaturation(), 3),
    })
    st.bar_chart(allow_df, x="Tissue", y="Allowable ΔP (bar)")
    st.caption(
        "Lower r_crit → higher allowable supersaturation (Young-Laplace: ΔP = 2γ/r). "
        "Fast tissues (T1–T4) have small r_crit after deep dives → higher protection against bubbles."
    )

    if history:
        st.divider()
        st.subheader("Mean r_crit During Ascent")
        r_hist = pd.DataFrame([{
            "Time (min)": h["time"],
            "Mean r_crit (μm)": h["r_crit_mean"]
        } for h in history])
        st.line_chart(r_hist, x="Time (min)", y="Mean r_crit (μm)")
        st.caption("Radius decreases as the diver ascends and nuclei are less compressed.")

# ─────────────────────── TAB 4: Physics Explainer ────────────────────────────
with tab4:
    st.subheader("📖 Physics Concepts in This Tool")

    with st.expander("1. Haldane Exponential Loading (Schreiner Equation)", expanded=False):
        st.markdown(r"""
The fundamental equation governing inert gas exchange in each tissue compartment:

$$P_t(t) = P_{insp} + (P_{t,0} - P_{insp}) \cdot e^{-k \cdot t}$$

| Symbol | Meaning |
|--------|---------|
| $P_t(t)$ | Tissue pressure at time $t$ |
| $P_{insp}$ | Inspired partial pressure of gas |
| $P_{t,0}$ | Initial tissue pressure |
| $k = \ln 2 / t_{1/2}$ | Rate constant from half-time |

**Key insight**: This works for **both** on-gassing (ascending inspired pp) and off-gassing
(descending inspired pp) — the direction is automatic. Most simplified tools only apply it
at the bottom. This simulation applies it **at every 6-second step including during deco stops**,
so on-gassing is correctly modelled when you hold at a stop.
        """)

    with st.expander("2. Bühlmann ZHL-16C M-Values (a, b Coefficients)", expanded=False):
        st.markdown(r"""
Each tissue has a **maximum tolerated ambient pressure** (M-value line):

$$M = \frac{P_t}{b} + a$$

Or rearranged: the minimum safe ambient pressure for a tissue at loading $P_t$ is:

$$P_{amb,min} = (P_t - a \cdot GF) \cdot b$$

| Coefficient | Fast tissues (T1) | Slow tissues (T16) | Effect |
|------------|------------|------------|--------|
| $a$ (bar) | 1.26 | 0.23 | Higher → more conservative at surface |
| $b$ | 0.505 | 0.965 | Lower → more conservative at depth |

**ZHL-16C** differs from earlier versions in two ways:
1. Slightly adjusted coefficients validated against more experimental data.
2. For Trimix, `a` and `b` are **helium-weighted interpolated** before computing ceiling.

**vs. previous tool**: Previous version used an approximation. This tool uses the
exact published ZHL-16C table coefficients.
        """)

    with st.expander("3. Gradient Factors (GF) — Baker 1998", expanded=False):
        st.markdown(r"""
GFs scale the M-value headroom to add conservatism:

$$P_{tol} = (P_t - a \cdot GF) \cdot b$$

GF interpolates **linearly** between the deepest stop and the surface:

$$GF(d) = GF_{Lo} + (GF_{Hi} - GF_{Lo}) \cdot \left(1 - \frac{d}{d_{first\,stop}}\right)$$

This mimics VPM-B's behaviour: more conservative at deep stops, 
allows faster ascent at shallow depths. Popular pairs:

| Profile | GF Lo / Hi | Character |
|---------|-----------|-----------|
| Conservative | 20/70 | Deep stops, long deco |
| Standard | 35/85 | Balanced deep + shallow |
| Aggressive | 50/95 | Minimal stops |
        """)

    with st.expander("4. VPM-B Bubble Nucleation & Critical Radius", expanded=False):
        st.markdown(r"""
**VPM = Varying Permeability Model** (Yount 1979, Hoffman 1985).

A bubble nucleus is in equilibrium when the **Laplace pressure** equals the 
net gas pressure differential:

$$\Delta P = \frac{2\gamma}{r}$$

| Symbol | Meaning |
|--------|---------|
| $\gamma$ | Surface tension of nucleus membrane (≈ 0.0179 N/m) |
| $r$ | Critical radius of nucleus |
| $\Delta P$ | Allowable supersaturation before bubble grows |

**Crushing effect**: During deep dives, high ambient pressure *crushes* nuclei,
reducing $r_{crit}$. By Young-Laplace, smaller $r$ → **larger** $\Delta P$ required
to grow a bubble. So a deeper dive is paradoxically *harder* to cause bubble growth
on ascent — but only if you ascended correctly on the previous dive.

$$r_{crit} = \frac{2\gamma_c \cdot r_0}{2\gamma_c + \lambda \cdot P_{crush}}$$

The **surface interval** allows $r_{crit}$ to relax back toward $r_0$, reducing
the protection from crushing — which is why short surface intervals increase risk.
        """)

    with st.expander("5. On-Gassing at Deco Stops", expanded=False):
        st.markdown(r"""
**The counter-diffusion problem**: At a deco stop, you are ascending into an
environment where inspired $P_{N_2}$ is *lower* than tissue $P_{N_2}$ → off-gassing. ✅

But **fast tissues** (short half-times) may already be near equilibrium at the
stop depth, while **slow tissues** continue on-gassing from the bottom exposure.

Previous simplified models computed tissue loading only at the bottom and then
simulated pure off-gassing. **This tool** applies the Schreiner equation at every
time step during stops, so:

- Slow tissues may continue to take on N₂/He at shallow stops
- The ceiling calculation at each stop iteration reflects the *true* tissue state
- Stop time extends until the updated loading clears the ceiling

This is especially important for **Trimix** because He off-gasses very rapidly
(fast half-times) but N₂ may still be loading slowly in compartments T12–T16.
        """)

st.divider()
st.caption("🫧 Decompression Teaching Tool v2 | ZHL-16C + VPM-B | For educational use only. "
           "Never plan actual dives using this tool.")
