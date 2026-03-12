import numpy as np
import streamlit as st
import math

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------

class DiveConfig:
    """Holds all physical and model constants."""
    def __init__(self):
        # ZHL-16 tissue compartment half-times (minutes)
        self.HALF_TIMES_N2 = np.array([4.0, 8.0, 12.5, 18.5, 27.0, 38.3, 54.3, 77.0,
                                       109.0, 146.0, 187.0, 239.0, 305.0, 390.0, 498.0, 635.0])
        self.HALF_TIMES_HE = np.array([1.51, 3.02, 4.72, 6.99, 10.21, 14.48, 20.53, 29.11,
                                       41.20, 55.19, 70.69, 90.34, 115.29, 147.42, 188.24, 240.03])
        
        # VPM-B Parameters
        self.CONSERVATISM_LEVEL = 2
        self.ASCENT_RATE = 9.0      # m/min
        self.SURFACE_PRESSURE = 1.013  # bar
        self.BAR_PER_METER = 0.098     # bar/m (seawater)
        self.PRESSURE_OTHER_GASES = 0.102 # bar (water vapor)
        
        # Pre-calculate rate constants (k)
        self.k_n2 = np.log(2) / self.HALF_TIMES_N2
        self.k_he = np.log(2) / self.HALF_TIMES_HE

    def get_vpm_gradients(self) -> np.ndarray:
        """
        Calculate the critical gradient (G_crit) for EACH tissue compartment.
        This represents the 'Bubble Growth Limit' in terms of pressure difference.
        Fast tissues have tighter limits (smaller gradient).
        """
        effective_ht = (self.HALF_TIMES_N2 + self.HALF_TIMES_HE) / 2
        
        # Formula: G_base = A + B / sqrt(ht)
        # Tuned to approximate VPM-B behavior: Fast tissues ~0.35 bar, Slow ~0.8 bar
        g_base = 0.35 + (0.65 / np.sqrt(effective_ht / 10.0))
        
        # Apply conservatism
        conservatism_factor = 1.0 - (0.05 * self.CONSERVATISM_LEVEL)
        
        return g_base * conservatism_factor

config = DiveConfig()

# -----------------------------------------------------------------------------
# Physics Engine
# -----------------------------------------------------------------------------

def round_up_to_3m(depth):
    """Rounds a depth up to the next deeper 3-meter increment."""
    if depth <= 0:
        return 0.0
    return math.ceil(depth / 3.0) * 3.0

def calculate_ambient_pressure(depth):
    return config.SURFACE_PRESSURE + depth * config.BAR_PER_METER

def calculate_inspired_partial_pressures(depth, fo2, fn2, fhe):
    ambient = calculate_ambient_pressure(depth)
    dry_pressure = ambient - config.PRESSURE_OTHER_GASES
    return (fo2 * dry_pressure, fn2 * dry_pressure, fhe * dry_pressure)

def load_tissues(depth, time, fo2, fn2, fhe, initial_n2=None, initial_he=None):
    """Exact exponential loading for all 16 compartments."""
    _, p_insp_n2, p_insp_he = calculate_inspired_partial_pressures(depth, fo2, fn2, fhe)
    
    if initial_n2 is None:
        initial_n2 = np.full(16, 0.79 * config.SURFACE_PRESSURE)
    if initial_he is None:
        initial_he = np.zeros(16)
        
    n2_load = p_insp_n2 + (initial_n2 - p_insp_n2) * np.exp(-config.k_n2 * time)
    he_load = p_insp_he + (initial_he - p_insp_he) * np.exp(-config.k_he * time)
    
    return n2_load, he_load

def calculate_bubble_limit_ceiling(n2_loading, he_loading, depth):
    """
    Calculates the ceiling based on the Bubble Growth Limit (Critical Gradient) per tissue.
    Returns: (ceiling_depth, leading_tissue_index)
    """
    ambient = calculate_ambient_pressure(depth)
    total_loading = n2_loading + he_loading
    
    # Get the 16 unique critical gradients (Bubble Growth Limits)
    g_crits = config.get_vpm_gradients()
    
    # Vectorized calculation: Req_Amb = P_tissue - G_crit
    # This is the ambient pressure required to keep bubbles stable
    req_amb = total_loading - g_crits
    
    # Calculate depth required for each tissue
    depths = np.maximum(0.0, (req_amb - config.SURFACE_PRESSURE) / config.BAR_PER_METER)
    
    max_ceiling = np.max(depths)
    
    if max_ceiling > 0.0:
        leading_tissue = int(np.argmax(depths))
    else:
        leading_tissue = -1
        
    return max_ceiling, leading_tissue

def simulate_ascent_full(max_depth, bottom_time, fo2, fn2, fhe):
    """
    Full simulation to find BOTH Off-gassing Start and Max Bubble Growth Limit.
    """
    # 1. Load tissues at bottom
    n2_load, he_load = load_tissues(max_depth, bottom_time, fo2, fn2, fhe)
    
    current_depth = max_depth
    dt = 0.1  # minutes (6 seconds)
    offgas_depth = None
    offgas_tissue = -1
    max_ceiling = 0.0
    ceiling_tissue = -1
    
    # Simulation loop
    while current_depth > 0.0:
        ambient = calculate_ambient_pressure(current_depth)
        total_load = n2_load + he_load
        supersaturation = total_load - ambient
        
        # Check Off-gassing (First time any tissue > ambient)
        if offgas_depth is None and np.any(supersaturation > 0):
            offgas_depth = current_depth
            offgas_tissue = int(np.argmax(supersaturation))
        
        # Check Bubble Growth Limit (Ceiling) with PER-TISSUE gradients
        current_ceiling, curr_leader = calculate_bubble_limit_ceiling(n2_load, he_load, current_depth)
        
        if current_ceiling > max_ceiling:
            max_ceiling = current_ceiling
            ceiling_tissue = curr_leader
            
        # Stop early if we passed both points
        if offgas_depth is not None and current_depth < (offgas_depth - 2.0) and current_depth < (max_ceiling - 1.0):
            break
            
        # Move up
        delta_d = config.ASCENT_RATE * dt
        current_depth = max(0.0, current_depth - delta_d)
        
        # Update tissues
        n2_load, he_load = load_tissues(current_depth, dt, fo2, fn2, fhe, initial_n2=n2_load, initial_he=he_load)
        
    return offgas_depth, offgas_tissue, max_ceiling, ceiling_tissue

# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------

st.set_page_config(page_title="Bubble Growth Limit Calculator", page_icon="🫧", layout="wide")

st.title("🫧 Teaching Tool: Bubble Growth Limit per Tissue")
st.markdown("### Understanding VPM-B Decompression Physics")
st.info("**Key Concept:** **Off-gassing** and **Ceiling.**")

# Sidebar
with st.sidebar:
    st.header("Dive Profile")
    fo2 = st.slider("Oxygen (%)", 0, 100, 25) / 100.0
    fhe = st.slider("Helium (%)", 0, 100, 25) / 100.0
    fn2 = 1.0 - fo2 - fhe
    
    if fn2 < 0:
        st.error("❌ Invalid Mix: O2 + He > 100%")
        st.stop()
    
    st.info(f"**Nitrogen:** {fn2*100:.1f}%")
    
    max_depth = st.number_input("Max Depth (m)", min_value=0.0, value=30.0, step=1.0)
    bottom_time = st.number_input("Bottom Time (min)", min_value=0.0, value=45.0, step=1.0)
    
    st.markdown("---")
    if st.button("Recalculate"):
        st.rerun()

# Main Calculation
if max_depth > 0:
    offgas_raw, offgas_idx, ceiling_raw, ceiling_idx = simulate_ascent_full(max_depth, bottom_time, fo2, fn2, fhe)
    
    # Calculate Practical Stops (Rounded to 3m)
    offgas_practical = round_up_to_3m(offgas_raw) if offgas_raw else 0.0
    ceiling_practical = round_up_to_3m(ceiling_raw) if ceiling_raw > 0 else 0.0
    
    # Calculate TRUE Safety Margin (Raw vs Raw)
    true_margin = (offgas_raw - ceiling_raw) if (offgas_raw and ceiling_raw > 0) else 0.0
    
    # Calculate "Rounding Buffer" (Extra safety added by the 3m rule)
    rounding_buffer = ceiling_practical - ceiling_raw if ceiling_practical > 0 else 0.0

    st.divider()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Off-Gassing Start", f"{offgas_raw:.1f} m" if offgas_raw else "Surface")
        st.caption(f"Practical: {offgas_practical:.0f} m")
        if offgas_idx != -1:
            st.caption(f"(Tissue {offgas_idx+1})")
    
    with col2:
        st.metric("Ceiling", f"{ceiling_raw:.1f} m" if ceiling_raw > 0 else "0.0 m")
        st.caption(f"Planned Stop: {ceiling_practical:.0f} m")
        if ceiling_idx != -1:
            st.caption(f"(Tissue {ceiling_idx+1})")
    
    with col3:
        if ceiling_raw > 0:
            st.metric("Safety Margin", f"{true_margin:.1f} m")
            st.caption(f"Plus {rounding_buffer:.1f}m from rounding")
        else:
            st.metric("Status", "No Deco")

    # Visual Explanation
    st.subheader("Ascent Profile & Stop Planning")
    
    if ceiling_practical > 0:
        st.warning(f"⚠️ **MANDATORY STOP**: Plan your first stop at **{ceiling_practical:.0f} meters**.")
        
        st.markdown(f"""
        ### Analysis:
        1. **Off-Gassing Begins**: At **{offgas_raw:.1f}m**, tissues start releasing gas safely.
        2. **Ceiling**: At **{ceiling_raw:.1f}m**, the release rate becomes dangerous (bubbles grow).
        3. **Safety Margin**: You have **{true_margin:.1f}m** of safe ascent zone between these points.
        
        
        """)
        
        # Visualization Chart
        chart_data = {
            'Zone': [
                'Surface (0m)', 
                f'Bubble Limit ({ceiling_raw:.1f}m)', 
                f'Planned Stop ({ceiling_practical:.0f}m)', 
                f'Off-Gassing Start ({offgas_raw:.1f}m)', 
                f'Bottom ({max_depth}m)'
            ],
            'Depth (m)': [0, ceiling_raw, ceiling_practical, offgas_raw, max_depth]
        }
        st.bar_chart(chart_data, x='Zone', y='Depth (m)', horizontal=True)
        
        st.success(f"✅ **Plan**: Ascend to **{ceiling_practical:.0f}m**. Stay there until bubble risk drops.")
        
    else:
        st.success("✅ **No Decompression Stop Required**.")
        st.markdown(f"Off-gassing begins at **{offgas_raw:.1f}m**. Bubble growth limit is not reached.")

    # Detailed Tissue Chart
    st.divider()
    st.subheader("Tissue Analysis at Planned Stop Depth")
    
    if ceiling_practical > 0:
        st.caption(f"Status at the planned stop depth of {ceiling_practical:.0f}m.")
        
        # Simulate to the PRACTICAL stop depth
        n2_load, he_load = load_tissues(max_depth, bottom_time, fo2, fn2, fhe)
        steps = int((max_depth - ceiling_practical) / 0.5)
        curr_d = max_depth
        for _ in range(steps):
            curr_d -= 0.5
            if curr_d < ceiling_practical: curr_d = ceiling_practical
            n2_load, he_load = load_tissues(curr_d, 0.05, fo2, fn2, fhe, initial_n2=n2_load, initial_he=he_load)
            
        total_load = n2_load + he_load
        g_crits = config.get_vpm_gradients()
        ambient_at_stop = calculate_ambient_pressure(ceiling_practical)
        limits = ambient_at_stop + g_crits
        
        chart_data = {
            'Tissue': [f"T{i+1}" for i in range(16)],
            'Inert Gas Pressure': total_load,
            'Bubble Growth Limit': limits
        }
        st.bar_chart(chart_data, x='Tissue', y=['Inert Gas Pressure', 'Bubble Growth Limit'])
        st.caption("Note: Because we rounded up, the Ambient Pressure at the stop is higher, pushing the Bubble Growth Limit (gray line) up and creating a larger safety gap.")

else:
    st.info("Please enter a depth greater than 0.")

st.markdown("---")
st.caption("Decompression Zone| Per-Tissue Bubble Limits | 3m Rounding Rule | Educational Use Only")
