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
    Runs the Decotengu simulation using the CORRECT functional API.
    Algorithm: ZHL-16B with Gradient Factors (to simulate a conservative ceiling).
    """
    try:
        # 1. Define Configuration
        # Decotengu uses a dictionary or specific config objects.
        # We set the algorithm to 'zhl16b' and define Gradient Factors.
        # GF Low (30%) creates the 'ceiling' (conservative stop).
        # GF High (85%) is the surface limit.
        config = {
            'algorithm': 'zhl16b',
            'gf_low': 0.30,
            'gf_high': 0.85,
            'step': 0.5,  # Simulation step in meters
        }
        
        # 2. Define Gas Mix
        # Format: (O2, He)
        gases = [(fo2, fhe)]
        
        # 3. Define Dive Profile
        # Format: [(depth, time), ...]
        # We define the bottom segment. The planner adds deco.
        # Descent/Ascent rates are handled by the planner defaults or args.
        profile = [(max_depth, bottom_time)]
        
        # --- Phase A: Get Plan (Stops & Ceiling) ---
        # dt.plan returns a Plan object
        # Arguments: gases, profile, config
        plan = dt.plan(gases, profile, config)
        
        max_ceiling = plan.ceiling
        stops = [{'depth': s.depth, 'time': s.time} for s in plan.stops]
        
        # --- Phase B: Simulate Ascent to find "Off-Gassing Start" ---
        # dt.simulate returns a list of dictionaries for each step.
        # Each dict contains: 'depth', 'time', 'tissues' (list of pressures)
        # We need to construct a full profile including ascent to simulate step-by-step.
        
        # Construct a profile that goes: Surface -> Bottom -> Surface
        # This allows us to inspect every step of the ascent.
        descent_rate = 22.0
        ascent_rate = 9.0
        
        descent_time = max_depth / descent_rate
        
        # We will manually build the simulation steps by calling dt.simulate 
        # on a profile that includes the ascent, OR we can step manually.
        # The cleanest way with Decotengu is to let it simulate the whole dive
        # and then iterate the result steps to find the first off-gassing point.
        
        # Create a "dummy" profile that forces an ascent to surface without stops
        # just to check tissue tensions? 
        # Actually, `dt.simulate` follows the plan. 
        # Let's simulate the planned dive. The result contains the ascent with stops.
        
        # dt.simulate returns a list of steps
        # We need to pass the plan to simulate? 
        # Actually, dt.simulate(gases, profile, config) runs the simulation.
        # It returns a list of steps.
        
        # Let's create a profile that represents the dive including the ascent.
        # But `dt.simulate` usually follows the deco plan automatically if passed to `dt.plan`.
        # Wait, `dt.simulate` is for open circuit simulation of a GIVEN profile.
        # `dt.plan` generates the stops.
        
        # To get tissue data during ascent:
        # We can run `dt.simulate` on the ORIGINAL profile (without stops) to see when tissues exceed ambient?
        # No, that would be a fatal dive simulation.
        # Better: Run `dt.simulate` on the COMPLETED plan (with stops).
        
        # The `plan` object has a `profile` attribute that includes the stops?
        # Let's try to simulate the planned profile.
        # In Decotengu, `plan.profile` gives the full profile with stops.
        
        full_profile = plan.profile
        
        # Now simulate this full profile to get tissue data at every step
        steps = dt.simulate(gases, full_profile, config)
        
        offgas_depth = None
        
        for step in steps:
            depth = step['depth']
            amb_bar = 1.013 + depth * 0.098
            tissues = step['tissues'] # List of inert gas pressures
            
            # Check if ANY tissue exceeds ambient
            # Note: tissues list contains N2 and He partial pressures for each compartment
            # We need to sum them per compartment.
            # The structure of `tissues` in Decotengu: 
            # It's a list of values. For trimix, it's [N2_1, He_1, N2_2, He_2, ...]?
            # Or is it list of tuples? 
            # Documentation says: "tissues: list of tissue pressures"
            # For multiple gases, it sums them internally? 
            # Let's assume standard ZHL output: list of total inert gas pressures per compartment.
            # If it returns separate N2/He, we must sum them.
            # Decotengu usually returns total inert gas pressure per compartment in the step dict 
            # IF configured correctly, OR we sum manually.
            
            # Safe check: iterate and sum if needed.
            # In standard ZHL16B implementation in Decotengu:
            # `tissues` is a list of length 16 (total inert gas pressure per compartment).
            
            is_offgassing = False
            for p_tissue in tissues:
                if p_tissue > amb_bar:
                    is_offgassing = True
                    break
            
            if is_offgassing and offgas_depth is None:
                # We found the first point during ascent (or descent/bottom) where off-gassing occurs.
                # But we only care about the ASCENT phase.
                # The steps list goes chronologically.
                # We need to ensure we are on the way up.
                # Simple check: if depth < max_depth and we haven't found it yet.
                # However, off-gassing can technically start on the bottom if time is long?
                # No, on the bottom P_tissue < P_insp = P_amb (approx).
                # So it only happens on ascent.
                offgas_depth = depth
                # We don't break immediately if we want the VERY first moment.
                # The loop is chronological, so the first hit is the start.
                break
        
        if offgas_depth is None:
            offgas_depth = 0.0
            
        return offgas_depth, max_ceiling, stops, plan
        
    except Exception as e:
        st.error(f"Simulation Error: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None, None, [], None

# --- STREAMLIT UI ---
st.set_page_config(page_title="ZHL-16B Dive Planner", page_icon="🤿", layout="wide")

st.title("🤿 Decompression Calculator (ZHL-16B)")
st.markdown("### Powered by **Decotengu** Library")
st.info("**Teaching Point:** **Off-gassing** (physics) happens when tissues exceed ambient pressure. The **Ceiling** (safety limit) depends on Gradient Factors.")

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
            3. **Safety Margin**: You have **{true_margin:.1f}m** of safe ascent zone.
            4. **The Rule**: We round the limit ({ceiling_raw:.1f}m) **UP** to **{ceiling_practical:.0f}m**.
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
        st.error("Failed to calculate profile.")

st.markdown("---")
st.caption("Powered by Decotengu | ZHL-16B Algorithm | Educational Use Only")
