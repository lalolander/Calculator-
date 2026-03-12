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

# --- STREAMLIT UI ---
st.set_page_config(page_title="VPM-B Dive Planner", page_icon="🤿", layout="wide")

st.title("🤿 VPM-B Decompression Calculator")
st.markdown("### Interactive Teaching Tool for Dive Physics")
st.markdown("Adjust the sliders to see how depth, time, and gas mix affect tissue loading and decompression ceilings.")

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
    st.caption("Algorithm: VPM-B (Per-Tissue Gradients)")

# Main Calculation
if max_depth > 0:
    # Initial Loading at Bottom
    n2_load, he_load = load_tissues(max_depth, bottom_time, fo2, fn2, fhe)
    
    # Simulate Ascent to find max ceiling
    current_depth = max_depth
    max_ceiling_found = 0.0
    leading_tissue_final = -1
    steps = int(max_depth / 0.5) + 1
    
    # We simulate the ascent to find the "worst case" ceiling
    temp_n2, temp_he = n2_load.copy(), he_load.copy()
    
    for _ in range(steps):
        c, l = calculate_vpm_ceiling_detailed(temp_n2, temp_he, current_depth)
        if c > max_ceiling_found:
            max_ceiling_found = c
            leading_tissue_final = l
        
        # Move up slightly
        step_dist = 0.5
        if current_depth > step_dist:
            current_depth -= step_dist
            # Update loading for this small step
            temp_n2, temp_he = load_tissues(current_depth, 0.05, fo2, fn2, fhe, initial_n2=temp_n2, initial_he=temp_he)
        else:
            break

    # Display Results
    st.divider()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Decompression Ceiling", f"{max_ceiling_found:.1f} m")
    with col2:
        if leading_tissue_final != -1:
            ht = config.HALF_TIMES_N2[leading_tissue_final]
            st.metric("Leading Tissue", f"T{leading_tissue_final+1} ({ht}m)")
        else:
            st.metric("Leading Tissue", "None")
    with col3:
        status = "⚠️ DECO REQUIRED" if max_ceiling_found > 0 else "✅ NO DECO"
        st.metric("Status", status)

    if max_ceiling_found > 0:
        st.warning(f"**STOP REQUIRED:** Do not ascend shallower than **{max_ceiling_found:.1f} meters**.")
        st.info(f"This stop is forced by **Tissue {leading_tissue_final+1}** (Half-time: {config.HALF_TIMES_N2[leading_tissue_final]} min N2).")
        
        # Visualization
        st.subheader("Tissue Loading Analysis")
        total_load = temp_n2 + temp_he
        g_crits = config.get_vpm_gradients()
        ambient_surface = config.SURFACE_PRESSURE
        limits = ambient_surface + g_crits
        
        chart_data = {
            'Tissue': [f"T{i+1}" for i in range(16)],
            'Current Load (bar)': total_load,
            'Critical Limit (bar)': limits
        }
        st.bar_chart(chart_data, x='Tissue', y=['Current Load (bar)', 'Critical Limit (bar)'])
        
        st.markdown("""
        **How to read this chart:**
        - **Blue Bars:** The amount of inert gas currently in your tissues.
        - **Gray Line:** The maximum safe limit for that tissue according to VPM-B.
        - **Red Zone:** Any blue bar exceeding the gray line means you must stop at depth to let that tissue off-gas safely.
        """)
    else:
        st.success("✅ You can ascend directly to the surface. (Safety stop at 5m for 3 mins still recommended).")
        
        # Show loading anyway for educational purposes
        st.subheader("Tissue Loading (No-Deco)")
        total_load = n2_load + he_load
        g_crits = config.get_vpm_gradients()
        limits = config.SURFACE_PRESSURE + g_crits
        chart_data = {
            'Tissue': [f"T{i+1}" for i in range(16)],
            'Current Load (bar)': total_load,
            'Critical Limit (bar)': limits
        }
        st.bar_chart(chart_data, x='Tissue', y=['Current Load (bar)', 'Critical Limit (bar)'])

else:
    st.info("Please enter a depth greater than 0 to begin.")

st.markdown("---")
st.caption("Created for Educational Purposes | VPM-B Algorithm Implementation")
