import streamlit as st
import decotengu as dt
import math

# --- CONFIGURATION & HELPERS ---

def round_up_to_3m(depth):
    """Rounds a depth up to the next deeper 3-meter increment."""
    if depth <= 0:
        return 0.0
    return math.ceil(depth / 3.0) * 3.0

def run_deco_simulation(fo2, fhe, max_depth, bottom_time):
    """
    Runs the Decotengu simulation using the Engine.
    Returns: offgas_depth, ceiling_depth, stops_list
    """
    try:
        # 1. Setup the Decompression Model
        # 'vpmb' is the string identifier for VPM-B in Decotengu
        model = dt.create_model('vpmb')
        
        # 2. Define Gas Mixes
        # Decotengu expects gases as a list of tuples or Gas objects: (O2, He)
        # N2 is automatically 1 - O2 - He
        gas_mix = (fo2, fhe)
        model.set_gases([gas_mix])
        
        # 3. Create the Dive Profile
        # We manually simulate the profile to extract "Off-gassing start"
        # Profile: Surface -> Descent -> Bottom -> Ascent
        
        descent_rate = 22.0 # m/min
        ascent_rate = 9.0   # m/min
        
        # Calculate times
        descent_time = max_depth / descent_rate
        
        # --- Phase A: Run the dive to get the Plan (Stops & Ceiling) ---
        # We use the high-level planner function included in the engine
        # dt.plan is a helper that returns a Plan object
        from decotengu import plan as dt_plan
        
        # Create a simple dive definition for the planner
        # Format: [(depth1, time1), (depth2, time2)...]
        # This represents the bottom profile only; the planner adds deco stops.
        dive_profile = [(max_depth, bottom_time)]
        
        # Run the planner
        # The planner returns a Plan object with .stops and .ceiling
        plan_result = dt_plan(model, dive_profile, descent_rate, ascent_rate)
        
        max_ceiling = plan_result.ceiling
        stops = [{'depth': s.depth, 'time': s.time} for s in plan_result.stops]
        
        # --- Phase B: Manual Simulation to find "Off-Gassing Start" ---
        # The planner doesn't explicitly return "off-gassing start depth".
        # We simulate the ascent step-by-step to find the exact moment tissues > ambient.
        
        # Reset model state for manual simulation
        model = dt.create_model('vpmb')
        model.set_gases([gas_mix])
        
        current_depth = 0.0
        current_time = 0.0
        
        # 1. Descent
        step = 0.5 # meters
        dt_step = step / descent_rate
        while current_depth < max_depth:
            current_depth += step
            if current_depth > max_depth: current_depth = max_depth
            model.step(dt_step, current_depth)
            
        # 2. Bottom Time
        # We can do this in one big step or small steps. One step is fine for loading.
        model.step(bottom_time, max_depth)
        
        # 3. Ascent (to find off-gassing)
        offgas_depth = None
        current_depth = max_depth
        
        while current_depth > 0:
            current_depth -= step
            if current_depth < 0: current_depth = 0
            
            dt_step = step / ascent_rate
            model.step(dt_step, current_depth)
            
            # Check tissues
            amb_bar = 1.013 + current_depth * 0.098
            is_offgassing = False
            
            # model.tissues contains the tissue objects
            for t in model.tissues:
                # t.p is the total inert gas pressure (N2 + He) in bar
                if t.p > amb_bar:
                    is_offgassing = True
                    break
            
            if is_offgassing and offgas_depth is None:
                offgas_depth = current_depth
                # We don't break immediately if we want to be super precise, 
                # but for teaching, the first detection is fine.
                break
        
        if offgas_depth is None:
            offgas_depth = 0.0
            
        return offgas_depth, max_ceiling, stops, plan_result
        
    except Exception as e:
        st.error(f"Simulation Error: {e}")
        import traceback
        st.code(traceback.format_exc())
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
                    'Depth (m)': [int(s['depth']) for s in stops], # Decotengu returns float, cast to int for clean display
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
