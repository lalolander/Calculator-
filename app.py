import numpy as np
import streamlit as st

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

# --- PHYSICS FUNCTIONS ---
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

st.title("🤿 VPM-B Decompression Calculator")
st.markdown("### Teaching Tool: Off-Gassing vs. Decompression Ceiling")
st.info("**Key Concept:** Off-gassing starts when tissues exceed ambient pressure. The **Ceiling** is where that off-gassing becomes dangerous (bubble growth).")

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
    offgas_depth, offgas_tissue_idx, ceiling_depth, ceiling_tissue_idx = simulate_ascent_full(max_depth, bottom_time, fo2, fn2, fhe)
    
    st.divider()
    
    # Display Metrics Side-by-Side
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Off-Gassing Starts", f"{offgas_depth:.1f} m" if offgas_depth else "Surface")
        if offgas_tissue_idx != -1:
            st.caption(f"(Tissue {offgas_tissue_idx+1})")
    
    with col2:
        st.metric("Decompression Ceiling", f"{ceiling_depth:.1f} m" if ceiling_depth > 0 else "0.0 m")
        if ceiling_tissue_idx != -1:
            st.caption(f"(Tissue {ceiling_tissue_idx+1})")
    
    with col3:
        diff = (offgas_depth - ceiling_depth) if offgas_depth and ceiling_depth > 0 else 0
        st.metric("Safety Margin", f"{diff:.1f} m")
        
    # Visual Explanation
    st.subheader("Ascent Profile Visualization")
    
    if ceiling_depth > 0:
        st.warning(f"⚠️ **STOP REQUIRED**: You must stop at **{ceiling_depth:.1f}m**.")
        st.markdown(f"""
        - You start off-gassing at **{offgas_depth:.1f}m**.
        - However, ascending shallower than **{ceiling_depth:.1f}m** is unsafe.
        - **Difference**: You must wait **{offgas_depth - ceiling_depth:.1f}m** deeper than the natural off-gassing point to prevent bubble growth.
        """)
        
        # Simple Chart of Depths
        chart_data = {
            'Zone': ['Surface', f'Ceiling ({ceiling_depth:.1f}m)', f'Off-Gassing Start ({offgas_depth:.1f}m)', f'Bottom ({max_depth}m)'],
            'Depth (m)': [0, ceiling_depth, offgas_depth, max_depth]
        }
        st.bar_chart(chart_data, x='Zone', y='Depth (m)', horizontal=True)
        
    else:
        st.success("✅ **No Decompression Stop Required**.")
        st.markdown(f"Off-gassing begins at **{offgas_depth:.1f}m**, but the gradient is safe all the way to the surface.")

    # Detailed Tissue Chart
    st.divider()
    st.subheader("Tissue Analysis at Ceiling Depth")
    
    # Recreate state at ceiling for the chart
    if ceiling_depth > 0 and offgas_depth:
        # Simulate down to ceiling to get exact loads
        n2_load, he_load = load_tissues(max_depth, bottom_time, fo2, fn2, fhe)
        # Quick step down to ceiling (approx)
        steps = int((max_depth - ceiling_depth) / 0.5)
        curr_d = max_depth
        for _ in range(steps):
            curr_d -= 0.5
            n2_load, he_load = load_tissues(curr_d, 0.05, fo2, fn2, fhe, initial_n2=n2_load, initial_he=he_load)
            
        total_load = n2_load + he_load
        g_crits = config.get_vpm_gradients()
        ambient_at_ceiling = calculate_ambient_pressure(ceiling_depth)
        limits = ambient_at_ceiling + g_crits
        
        chart_data = {
            'Tissue': [f"T{i+1}" for i in range(16)],
            'Inert Gas Pressure': total_load,
            'Critical Limit': limits
        }
        st.bar_chart(chart_data, x='Tissue', y=['Inert Gas Pressure', 'Critical Limit'])
        st.caption("Blue bars above the gray line indicate tissues forcing the decompression stop.")

else:
    st.info("Please enter a depth greater than 0.")

st.markdown("---")
st.caption("VPM-B Algorithm | Per-Tissue Gradients | Educational Use Only")
