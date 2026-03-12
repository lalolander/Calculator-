import streamlit as st
import decotengu as dt
import math

# --- CONFIGURATION & HELPERS ---

def round_up_to_3m(depth):
    """Rounds a depth up to the next deeper 3-meter increment."""
    if depth <= 0:
        return 0.0
    return math.ceil(depth / 3.0) * 3.0

def get_dive_profile(max_depth, bottom_time):
    """Creates a Decotengu dive profile object."""
    # Create a dive object
    dive = dt.Dive()
    # Add a descent, bottom time, and ascent
    # Descent rate: 22 m/min (standard)
    # Ascent rate: 9 m/min (standard)
    dive.add_segment(max_depth, bottom_time, descent_rate=22, ascent_rate=9)
    return dive

def run_deco_simulation(fo2, fhe, max_depth, bottom_time):
    """
    Runs the Decotengu simulation and extracts key teaching metrics.
    Returns: offgas_depth, ceiling_depth, stops_list
    """
    try:
        # Setup planner
        planner = dt.Planner()
        
        # Select algorithm: 'zhl16b' or 'vpmb' (VPM-B is what we used before)
        # Decotengu uses 'vpmb' for VPM-B
        planner.algorithm = 'vpmb' 
        
        # Setup gas mix
        # Decotengu expects mix as (O2, He) fractions. N2 is calculated automatically.
        gas_mix = dt.Gas(fo2, fhe)
        planner.gases = [gas_mix]
        
        # Create dive
        dive = get_dive_profile(max_depth, bottom_time)
        
        # Plan the dive
        # This returns a Plan object containing all stops and ceiling info
        plan = planner.plan(dive)
        
        # Extract Data for Teaching
        # 1. Max Ceiling (The deepest required stop during ascent)
        # Decotengu plan object has a 'ceiling' attribute that updates per segment
        # We need to find the maximum ceiling encountered during the ascent simulation
        max_ceiling = 0.0
        
        # Decotengu plan contains segments. We iterate to find the highest ceiling.
        # Note: In VPM-B, the ceiling is often highest right at the end of the bottom time 
        # or early in the ascent.
        if hasattr(plan, 'ceiling'):
            max_ceiling = plan.ceiling
            
        # 2. Off-gassing Start Depth (Approximation)
        # Decotengu doesn't explicitly output "off-gassing start" as a single metric 
        # because it focuses on the ceiling. 
        # However, we can approximate it: Off-gassing starts when Tissue Pressure > Ambient.
        # In Decotengu, this usually happens slightly above the max ceiling or at the same depth.
        # For teaching purposes with Decotengu, we can estimate:
        # If there is a ceiling, off-gassing definitely started deeper.
        # A safe approximation for the "Start of Off-gassing" in VPM-B is often 
        # the depth where the first tissue becomes supersaturated.
        # Since Decotengu hides the raw tissue tensions in the simple plan, 
        # we will use a heuristic: Off-gassing starts ~3-5m deeper than the ceiling for typical dives,
        # OR we can look at the first stop.
        # BETTER APPROACH FOR TEACHING: 
        # We will simulate the ascent step-by-step using the low-level API to find the exact moment.
        
        offgas_depth = None
        
        # Low-level simulation to find exact off-gassing start
        # We manually step the ascent to check tissue tensions
        model = dt.create_model('vpmb')
        model.set_gases([gas_mix])
        
        # Run bottom phase
        current_time = 0.0
        current_depth = 0.0
        
        # Descent
        descent_time = max_depth / 22.0
        # We skip detailed check on descent as on-gassing happens
        
        # Bottom phase start
        current_time += descent_time
        current_depth = max_depth
        
        # Run bottom segment to load tissues
        # We need the state of the model after the bottom time
        # Decotengu 'plan' already did this, but we need internal state.
        # Let's use the planner's internal model state if accessible, 
        # or re-simulate simply.
        
        # Re-simulate for teaching metrics:
        m = dt.create_model('vpmb')
        m.set_gases([gas_mix])
        
        # Descent
        m.step(descent_time, max_depth) # Simplified step
        
        # Bottom time simulation (step by step to be precise? No, one big step is fine for loading)
        m.step(bottom_time, max_depth)
        
        # Ascent simulation to find off-gassing
        # Step up in small increments
        step_dist = 0.5 # meters
        ascent_rate = 9.0 # m/min
        dt_step = step_dist / ascent_rate # time per step
        
        current_d = max_depth
        while current_d > 0:
            # Move up
            current_d -= step_dist
            if current_d < 0: current_d = 0
            
            # Step the model
            m.step(dt_step, current_d)
            
            # Check tissue tensions
            # m.tissues gives us the current inert gas pressure in each compartment
            # Ambient pressure
            amb_bar = 1.013 + current_d * 0.098
            
            # Check if ANY tissue exceeds ambient
            # m.tissues is a list of tissue objects. .p is the pressure.
            is_offgassing = False
            for t in m.tissues:
                # t.p is total inert gas pressure (N2 + He)
                if t.p > amb_bar:
                    is_offgassing = True
                    break
            
            if is_offgassing and offgas_depth is None:
                offgas_depth = current_d
                break # Found it
        
        if offgas_depth is None:
            offgas_depth = 0.0
            
        # 3. Decompression Stops
        stops = []
        if hasattr(plan, 'stops'):
            for stop in plan.stops:
                stops.append({'depth': stop.depth, 'time': stop.time})
                
        return offgas_depth, max_ceiling, stops, plan
        
    except Exception as e:
        st.error(f"Simulation Error: {e}")
        return None, None, [], None

# --- STREAMLIT UI ---
st.set_page_config(page_title="VPM-B Dive Planner (Decotengu)", page_icon="🤿", layout="wide")

st.title("🤿 Professional VPM-B Decompression Calculator")
st.markdown("### Powered by **Decotengu** Library")
st.info("**Teaching Point:** Compare the **Theoretical Off-Gassing Depth** (when gas starts leaving) with the **Decompression Ceiling** (when it becomes dangerous).")

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
    offgas_raw, ceiling_raw, stops, plan_obj = run_deco_simulation(fo2, fhe, max_depth, bottom_time)
    
    if offgas_raw is not None:
        # Apply 3m rounding
        offgas_practical = round_up_to_3m(offgas_raw) if offgas_raw > 0 else 0.0
        ceiling_practical = round_up_to_3m(ceiling_raw) if ceiling_raw > 0 else 0.0
        
        # True Margin
        true_margin = (offgas_raw - ceiling_raw) if (offgas_raw and ceiling_raw > 0) else 0.0
        rounding_buffer = ceiling_practical - ceiling_raw if ceiling_practical > 0 else 0.0

        st.divider()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Off-Gassing Start", f"{offgas_raw:.1f} m")
            st.caption(f"Practical: {offgas_practical:.0f} m")
        
        with col2:
            st.metric("Theoretical Ceiling", f"{ceiling_raw:.1f} m")
            st.caption(f"Planned Stop: {ceiling_practical:.0f} m")
        
        with col3:
            if ceiling_raw > 0:
                st.metric("True Safety Margin", f"{true_margin:.1f} m")
                st.caption(f"Plus {rounding_buffer:.1f}m from rounding")
            else:
                st.metric("Status", "No Deco")

        # Visual Explanation
        st.subheader("Ascent Profile & Stop Planning")
        
        if ceiling_practical > 0:
            st.warning(f"⚠️ **MANDATORY STOP**: Plan your first stop at **{ceiling_practical:.0f} meters**.")
            
            st.markdown(f"""
            ### Analysis (Decotengu VPM-B):
            1. **Off-Gassing Begins**: At **{offgas_raw:.1f}m**, tissues exceed ambient pressure.
            2. **The Limit**: At **{ceiling_raw:.1f}m**, bubble growth becomes critical.
            3. **True Margin**: You have **{true_margin:.1f}m** of safe ascent zone.
            4. **The Rule**: We round the limit ({ceiling_raw:.1f}m) **UP** to **{ceiling_practical:.0f}m**.
            5. **Result**: Your stop is **{rounding_buffer:.1f}m deeper** than the theoretical limit.
            """)
            
            # Chart
            chart_data = {
                'Zone': [
                    'Surface (0m)', 
                    f'Theoretical Ceiling ({ceiling_raw:.1f}m)', 
                    f'Planned Stop ({ceiling_practical:.0f}m)', 
                    f'Off-Gassing Start ({offgas_raw:.1f}m)', 
                    f'Bottom ({max_depth}m)'
                ],
                'Depth (m)': [0, ceiling_raw, ceiling_practical, offgas_raw, max_depth]
            }
            st.bar_chart(chart_data, x='Zone', y='Depth (m)', horizontal=True)
            
            # Detailed Stops Table
            if stops:
                st.subheader("Decompression Schedule")
                stop_df = {
                    'Depth (m)': [s['depth'] for s in stops],
                    'Time (min)': [f"{s['time']:.1f}" for s in stops]
                }
                st.table(stop_df)
                
        else:
            st.success("✅ **No Decompression Stop Required**.")
            st.markdown(f"Off-gassing begins at **{offgas_raw:.1f}m**. Safe to surface.")

    else:
        st.error("Failed to calculate profile. Please check inputs.")

st.markdown("---")
st.caption("Powered by Decotengu | VPM-B Algorithm | Educational Use Only")
