import numpy as np
import streamlit as st
import math

# --- CONFIGURATION CLASS ---
class DiveConfig:
    def __init__(self):
        self.HALF_TIMES_N2 = np.array([4.0, 8.0, 12.5, 18.5, 27.0, 38.3, 54.3, 77.0,
                                       109.0, 146.0, 187.0, 239.0, 305.0, 390.0, 498.0, 635.0])
        self.HALF_TIMES_HE = np.array([1.51, 3.02, 4.72, 6.99, 10.21, 14.48, 20.53, 29.11,
                                       41.20, 55.19, 70.69, 90.34, 115.29, 147.42, 188.24, 240.03])
        self.CONSERVATISM_LEVEL = 2
        self.ASCENT_RATE = 9.0
        self.SURFACE_PRESSURE = 1.013
        self.BAR_PER_METER = 0.098
        self.PRESSURE_OTHER_GASES = 0.102
        self.k_n2 = np.log(2) / self.HALF_TIMES_N2
        self.k_he = np.log(2) / self.HALF_TIMES_HE

    def get_vpm_gradients(self) -> np.ndarray:
        effective_ht = (self.HALF_TIMES_N2 + self.HALF_TIMES_HE) / 2
        g_base = 0.35 + (0.65 / np.sqrt(effective_ht / 10.0))
        conservatism_factor = 1.0 - (0.05 * self.CONSERVATISM_LEVEL)
        return g_base * conservatism_factor

config = DiveConfig()

# --- HELPER FUNCTIONS ---

def round_up_to_3m(depth):
    """
    Rounds a depth up to the next deeper 3-meter increment.
    Standard stops: 3, 6, 9, 12, 15, 18, 21...
    """
    if depth <= 0:
        return 0.0
    # Divide by 3, ceil to next integer, multiply by 3
    return math.ceil(depth / 3.0) * 3.0

def calculate_ambient_pressure(depth):
    return config.SURFACE_PRESSURE + depth * config.BAR_PER_METER

def load_tissues(depth, time, fo2, fn2, fhe, initial_n2=None, initial_he=None):
    ambient = calculate_ambient_pressure(depth)
    dry_pressure = ambient - config.PRESSURE_OTHER_GASES
    p_insp_n2, p_insp_he = fn2 * dry_pressure, fhe * dry_pressure
    
    if initial_n2 is None:
        initial_n2 = np.full(16, 0.79 * config.SURFACE_PRESSURE)
    if initial_he is None:
        initial_he = np.zeros(16)
        
    n2_load = p_insp_n2 + (initial_n2 - p_insp_n2) * np.exp(-config.k_n2 * time)
    he_load = p_insp_he + (initial_he - p_insp_he) * np.exp(-config.k_he * time)
    return n2_load, he_load

def calculate_vpm_ceiling_detailed(n2_loading, he_loading, depth):
    ambient = calculate_ambient_pressure(depth)
    total_loading = n2_loading + he_loading
    g_crits = config.get_vpm_gradients()
    req_amb = total_loading - g_crits
    depths = np.maximum(0.0, (req_amb - config.SURFACE_PRESSURE) / config.BAR_PER_METER)
    max_ceiling = np.max(depths)
    leader = int(np.argmax(depths)) if max_ceiling > 0.0 else -1
    return max_ceiling, leader

def simulate_ascent_full(max_depth, bottom_time, fo2, fn2, fhe):
    """
    Full simulation to find BOTH Off-gassing Start and Max Ceiling.
    """
    # 1. Load at bottom
    n2_load, he_load = load_tissues(max_depth, bottom_time, fo2, fn2, fhe)
    
    current_depth = max_depth
    dt = 0.1  # min
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
        
        # Check Ceiling
        current_ceiling, curr_leader = calculate_vpm_ceiling_detailed(n2_load, he_load, current_depth)
        if current_ceiling > max_ceiling:
            max_ceiling = current_ceiling
            ceiling_tissue = curr_leader
            
        # Stop early if we passed both points
        if offgas_depth is not None and current_depth < (offgas_depth - 2.0) and current_depth < (max_ceiling - 1.0):
            break
            
        # Move up
        delta_d = config.ASCENT_RATE * dt
        current_depth = max(0.0, current_depth - delta_d)
        n2_load, he_load = load_tissues(current_depth, dt, fo2, fn2, fhe, initial_n2=n2_load, initial_he=he_load)
        
    return offgas_depth, offgas_tissue, max_ceiling, ceiling_tissue

# --- STREAMLIT UI ---
st.set_page_config(page_title="VPM-B Dive Planner", page_icon="🤿", layout="wide")

st.title("🤿 Off-gassing Decompression Calculator by Carlos Lander")
st.markdown("### Teaching Tool: Theory vs. Practical Stops")
st.info("**Rule:** Decompression stops are always rounded **UP** to the next deeper **3-meter increment** (3, 6, 9, 12...) to ensure a safety margin.")

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
    offgas_depth_raw, offgas_tissue_idx, ceiling_raw, ceiling_tissue_idx = simulate_ascent_full(max_depth, bottom_time, fo2, fn2, fhe)
    
    # Apply the 3m rounding rule
    offgas_practical = round_up_to_3m(offgas_depth_raw) if offgas_depth_raw else 0.0
    ceiling_practical = round_up_to_3m(ceiling_raw) if ceiling_raw > 0 else 0.0
    
    st.divider()
    
    # Display Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Theoretical Off-Gassing", f"{offgas_depth_raw:.1f} m" if offgas_depth_raw else "Surface")
        st.caption(f"Practical: {offgas_practical:.0f} m")
    
    with col2:
        st.metric("Theoretical Ceiling", f"{ceiling_raw:.1f} m" if ceiling_raw > 0 else "0.0 m")
        st.caption(f"Practical Stop: {ceiling_practical:.0f} m")
    
    with col3:
        if ceiling_practical > 0:
            margin = offgas_practical - ceiling_practical
            st.metric("Safety Margin", f"{margin:.0f} m")
        else:
            st.metric("Status", "No Deco")

    # Visual Explanation
    st.subheader("Ascent Profile & Stop Planning")
    
    if ceiling_practical > 0:
        st.warning(f"⚠️ **MANDATORY STOP**: Plan your first stop at **{ceiling_practical:.0f} meters**.")
        st.markdown(f"""
        - **Theory**: Your tissues force a ceiling at **{ceiling_raw:.1f}m**.
        - **Practice**: We round up to the next 3m increment → **{ceiling_practical:.0f}m**.
        - **Off-Gassing**: Starts at **{offgas_depth_raw:.1f}m** (Rounded: {offgas_practical:.0f}m).
        - **Result**: You will hold at {ceiling_practical:.0f}m until your tissues allow you to ascend to the next stop ({ceiling_practical - 3:.0f}m).
        """)
        
        # Visualization Chart
        # Create a list of standard stops for the chart context
        max_stop = max(ceiling_practical, offgas_practical, max_depth)
        stops = list(range(0, int(max_stop) + 4, 3))
        
        chart_data = {
            'Reference Points': ['Surface (0m)', f'First Stop ({ceiling_practical:.0f}m)', f'Off-Gassing Start ({offgas_practical:.0f}m)', f'Max Depth ({max_depth}m)'],
            'Depth (m)': [0, ceiling_practical, offgas_practical, max_depth]
        }
        st.bar_chart(chart_data, x='Reference Points', y='Depth (m)', horizontal=True)
        
        st.success(f"✅ **Plan**: Ascend to **{ceiling_practical:.0f}m**. Stay there until the ceiling drops to {ceiling_practical - 3:.0f}m.")
        
    else:
        st.success("✅ **No Decompression Stop Required**.")
        st.markdown(f"Off-gassing begins at **{offgas_depth_raw:.1f}m**. You can ascend directly to the surface, but a safety stop at **3m** is recommended.")

    # Detailed Tissue Chart
    st.divider()
    st.subheader("Tissue Analysis at Practical Stop Depth")
    
    if ceiling_practical > 0:
        st.caption(f"Showing tissue status at the planned stop depth of {ceiling_practical:.0f}m.")
        # Simulate down to the PRACTICAL stop depth
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
            'Critical Limit': limits
        }
        st.bar_chart(chart_data, x='Tissue', y=['Inert Gas Pressure', 'Critical Limit'])
        st.caption("At the rounded stop depth, the ambient pressure is slightly higher than the theoretical ceiling, providing a safety buffer.")

else:
    st.info("Please enter a depth greater than 0.")

st.markdown("---")
st.caption("VPM-B Algorithm | 3-Meter Increment Rounding | Educational Use Only")
