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
    Runs the Decotengu simulation using ZHL-16B.
    We use Gradient Factors to simulate a 'Ceiling' (conservative) vs 'Theoretical Limit'.
    Returns: offgas_depth, ceiling_depth, stops_list
    """
    try:
        # 1. Setup the Decompression Model (ZHL-16B)
        # Decotengu uses ZhlModel. We will apply Gradient Factors later.
        model = dt.ZhlModel()
        
        # 2. Define Gas Mixes
        gas_mix = (fo2, fhe)
        model.set_gases([gas_mix])
        
        # 3. Define Gradient Factors (GF)
        # GF Low (e.g., 0.30): Simulates a conservative "VPM-like" ceiling.
        # GF High (e.g., 0.85): Simulates the theoretical max limit.
        # For this teaching tool, we fix GF Low to find the "Mandatory Stop".
        gf_low = 0.30
        gf_high = 0.85
        model.set_gradient_factors(gf_low, gf_high)
        
        # 4. Create the Dive Profile
        descent_rate = 22.0
        ascent_rate = 9.0
        
        # --- Phase A: Run the planner to get Stops & Ceiling ---
        from decotengu import plan as dt_plan
        
        dive_profile = [(max_depth, bottom_time)]
        
        # Run the planner with the set Gradient Factors
        plan_result = dt_plan(model, dive_profile, descent_rate, ascent_rate)
        
        # The 'ceiling' in the plan result is the deepest stop required based on GF Low
        max_ceiling = plan_result.ceiling
        stops = [{'depth': s.depth, 'time': s.time} for s in plan_result.stops]
        
        # --- Phase B: Manual Simulation to find "Off-Gassing Start" ---
        # Off-gassing is purely physical: Tissue Pressure > Ambient.
        # It does NOT depend on Gradient Factors.
        
        sim_model = dt.ZhlModel()
        sim_model.set_gases([gas_mix])
        # No gradient factors needed for off-gassing detection
        
        current_depth = 0.0
        step = 0.5 # meters
        
        # 1. Descent
        dt_step = step / descent_rate
        while current_depth < max_depth:
            current_depth += step
            if current_depth > max_depth: current_depth = max_depth
            sim_model.step(dt_step, current_depth)
            
        # 2. Bottom Time
        sim_model.step(bottom_time, max_depth)
        
        # 3. Ascent (to find off-gassing)
        offgas_depth = None
        current_depth = max_depth
        
        while current_depth > 0:
            current_depth -= step
            if current_depth < 0: current_depth = 0
            
            dt_step = step / ascent_rate
            sim_model.step(dt_step, current_depth)
            
            # Check tissues
            amb_bar = 1.013 + current_depth * 0.098
            is_offgassing = False
            
            for t in sim_model.tissues:
                # t.p is total inert gas pressure
                if t.p > amb_bar:
                    is_offgassing = True
                    break
            
            if is_offgassing and offgas_depth is None:
                offgas_depth = current_depth
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
st.set_page_config(page_title="ZHL-16B Dive Planner (Decotengu)", page_icon="🤿", layout="wide")

st.title("🤿 Decompression Calculator (ZHL-16B)")
st.markdown("### Powered by **Decotengu** Library")
st.info("**Teaching Point:** **Off-gassing** (physics) happens when tissues exceed ambient pressure. The **Ceiling** (safety limit) depends on the algorithm's conservatism (Gradient Factors).")

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
    st.caption("Algorithm: ZHL-16B | GF: 30/85")
    
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
                st.metric("Safety Margin", f"{true_margin:.1f} m")
                st.caption(f"Plus {rounding_buffer:.1f}m from rounding")
            else:
                st.metric("Status", "No Deco")

        # Visual Explanation
        st.subheader("Ascent Profile & Stop Planning")
        
        if ceiling_practical > 0:
            st.warning(f"⚠️ **MANDATORY STOP**: Plan your first stop at **{ceiling_practical:.0f} meters**.")
            
            st.markdown(f"""
            ### Analysis (ZHL-16B with Gradient Factors):
            1. **Off-Gassing Begins**: At **{offgas_raw:.1f}m**, tissues exceed ambient pressure (Physics).
            2. **The Limit**: At **{ceiling_raw:.1f}m**, the gradient exceeds the safe limit (Algorithm).
            3. **Safety Margin**: You have **{true_margin:.1f}m** of safe ascent zone between these points.
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
                    'Depth (m)': [int(s['depth']) for s in stops],
                    'Time (min)': [f"{s['time']:.1f}" for s in stops]
                }
                st.table(stop_df)
                
        else:
            st.success("✅ **No Decompression Stop Required**.")
            st.markdown(f"Off-gassing begins at **{offgas_raw:.1f}m**. Safe to surface.")

    else:
        st.error("Failed to calculate profile. Please check inputs.")

st.markdown("---")
st.caption("Powered by Decotengu | ZHL-16B Algorithm | Educational Use Only")
