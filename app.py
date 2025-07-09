import streamlit as st
import json
import subprocess
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# --- Página principal con tabs ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
st.set_page_config(layout="wide", page_title="R3Group Supply Chain Proof of Concept")

st.markdown(
    """
    <div style="display: flex; align-items: right; justify-content: space-between; padding: 10px 0;">
        <img src="http://r3group-project.com/wp-content/uploads/2023/04/REGROUP-Logo-project.png" alt="Logo" style="height: 60px;">
    </div>
    """,
    unsafe_allow_html=True
)



# --- Menú superior tipo tabs ---
tabs = st.tabs(["🏠 Home","Prediction", "📊 Simulation", "⚙️ Optimization"])

# --- Home Page ---
with tabs[0]:
    st.title("Supply Chain Simulation App")
   # st.image("logo.png", width=200)  # Cambia a tu logo
    st.markdown("""Welcome to the GLN supply chain twin digital tool.
This application allows you to enter parameters, launch simulations, and compare results between scenarios.
    
    """)

# --- Prediction ---
with tabs[1]:
    st.title("Price prediction")
   # st.image("logo.png", width=200)  # Cambia a tu logo

   #Row 1: input
    st.subheader("Input parameters")
   # --- Carpeta temporal ---
    temp_dir = os.path.join(BASE_DIR, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    # --- Carga de ficheros ---
    raw_material_prices_file = st.file_uploader(label="Please, select the time series raw material prices.")
    main_material_prices_file = st.file_uploader(label="Please, select the first supplier material prices.")

    # Variables para guardar las rutas
    raw_material_prices_path = None
    main_material_prices_path = None

    # Si se sube el primero
    if raw_material_prices_file is not None:
        raw_material_prices_path = os.path.join(temp_dir, raw_material_prices_file.name)
        with open(raw_material_prices_path, "wb") as f:
            f.write(raw_material_prices_file.getbuffer())
        st.success(f"Raw material prices saved to {raw_material_prices_path}")

    # Si se sube el segundo
    if main_material_prices_file is not None:
        main_material_prices_path = os.path.join(temp_dir, main_material_prices_file.name)
        with open(main_material_prices_path, "wb") as f:
            f.write(main_material_prices_file.getbuffer())
        st.success(f"Main supplier prices saved to {main_material_prices_path}")
    
    if st.button("Start prediction"):
        st.write(main_material_prices_path)
        result = subprocess.run(
            [sys.executable, os.path.join(BASE_DIR, "GRAL_forecasting_v1.1.py"), raw_material_prices_path, main_material_prices_path],
            capture_output=True,
            text=True
    )
   #Row 2: col 1= graph forecasting row material; col 2 = graph with PBT forecasted prices
   # Layout 2: Graphs
    st.markdown("---")
    st.subheader("Scenarios comparison")
    # 
    top_col1, top_col2 = st.columns([1, 1])
    with top_col1:
        st.subheader("Raw material prices forecast")
        output_file = os.path.join(BASE_DIR, "output", "raw_material_prices.png")
        if os.path.exists(output_file):
            st.image(output_file, caption="Historical and forecasted raw materials price")

    with top_col2:
        st.subheader("Supplier material vs raw material prices")
        output_file = os.path.join(BASE_DIR, "output", "predictions.png")
        if os.path.exists(output_file):
            st.image(output_file, caption="Supplier vs Raw materials price forecast")

 

# --- Simulation Page ---
with tabs[2]:
   

    #Run AS IS Scenario
    st.session_state.simulation_done = False
    st.header("Scenario simulation")
    param1="AS_IS"
    param2=0
    param3=-1
    param4=-1
    params = {
                "SimulationName": param1,
                "DemandVariation": param2,
                "InitWeekShortage": param3,
                "EndWeekShortage": param4
            }
    input_file = os.path.join(BASE_DIR, "input", "main_parameters.json")
    with open(input_file, "w") as f:
        json.dump(params, f)

    # Row 1: Input + Graphs
    top_col1, top_col2, top_col3 = st.columns([1, 2, 2])

    # Layout 1: Inputs
    param1 ="STRESS_TEST"
    with top_col1:
        st.subheader("Input parameters: STRESS TEST")       
        param2 = st.number_input("Demand variation (-1.0, 1.0)", value=0.0)
        param3 = st.number_input("Week starting the supply shortage", value=-1)
        param4 = st.number_input("Week ending the supply shortage", value=-1)

        if st.button("Start simulation"):
            params = {
                "SimulationName": param1,
                "DemandVariation": param2,
                "InitWeekShortage": param3,
                "EndWeekShortage": param4
            }
            input_file = os.path.join(BASE_DIR, "input", "main_parameters.json")
            with open(input_file, "w") as f:            
                json.dump(params, f)

            result = subprocess.run(
                [sys.executable, os.path.join(BASE_DIR, "GLN_simulation.py")],
                capture_output=True,
                text=True
            )
            
            st.session_state.simulation_done = True
            st.session_state.simulation_output = result.stdout            

        if st.button("Clean"):
            st.session_state.simulation_done = False
            st.experimental_rerun()

    # Layout 2: Graphs
    
    if  st.session_state.simulation_done==True:
        st.success("Simulation completed")
        st.text(st.session_state.simulation_output)
        with top_col2:
            st.subheader("AS IS")
            output_file_1 = os.path.join(BASE_DIR, "output", "AS_IS_individual.png")
            if os.path.exists(output_file_1):
                st.image(output_file_1, caption="AS-IS")
            output_file_2 = os.path.join(BASE_DIR, "output", "AS_IS_combined.png")
            if os.path.exists(output_file_1):
                st.image(output_file_2, caption="AS-IS")

        with top_col3:
            st.subheader("Simulation Scenario")
            output_file_1 = os.path.join(BASE_DIR, "output", "STRESS_TEST_individual.png")
            if os.path.exists( output_file_1):
                st.image(output_file_1, caption=param1)
            output_file_2 = os.path.join(BASE_DIR, "output", "STRESS_TEST_combined.png")
            if os.path.exists(output_file_1):
                st.image(output_file_2, caption=param1)

    # Row 2: KPI Table (full width)st.markdown("---")


        st.markdown("---")
        st.subheader("Scenarios comparison")

        output_file = os.path.join(BASE_DIR, "output", "comparison.json")
        if os.path.exists(output_file):
            with open(output_file) as f:
                df = json.load(f)

            scenarios = list(df.keys())   # Aquí ya te da ['AS_IS_vs_STRESS_TEST']

            # Seleccionar el último escenario
            last_scenario = scenarios[-1]

            # Extraer los KPIs del último escenario como lista de (key, value)
            items = list(df[last_scenario].items())   # OJO aquí es df[last_scenario]
      

            # Scenarios characteristics: primeros 3 KPIs
            characteristics = pd.DataFrame(items[0:3], columns=["KPI", "Value"])

            # Financial KPIs: 4º, 5º y 6º KPI
            financial_kpis = pd.DataFrame(items[3:6], columns=["KPI", "Value"])
            
            # Operational KPIs: últimos 7 KPIs
            operational_kpis = pd.DataFrame(items[-7:], columns=["KPI", "Value"])

            # Mostrar en Streamlit
            #st.subheader(f"Characteristics for: {last_scenario}")
            #st.dataframe(characteristics)

            st.subheader(f"Financial KPIs for: {last_scenario}")
            st.dataframe(financial_kpis)

            st.subheader(f"Operational KPIs for: {last_scenario}")
            st.dataframe(operational_kpis)

        # --- Optimization  Page ---
with tabs[3]:
    st.header("Optimization")
    #Row 1
    st.title(f"Under construction")
   # st.text(f"Recalculate batch size quantitiy, minimum order quantity or both")
   # st.text(f"Once calculated: simulate and compare the impact of demand variation with the regular BSQ, with the regular MOQ or both")
   # st.title(f"defining the logic and code changes")
    #st.text(f"first, ZLC dicuss internally if possible to host the app and give access to specific users")
   # st.text(f"In parallel, start deploying the app in R3Group platform")






st.markdown(
"""
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f0f2f6;
        color: #333;
        text-align: center;
        padding: 10px;
        font-size: 13px;
        border-top: 1px solid #ccc;
        z-index: 100;
    }
    .footer img {
        height: 40px;
        vertical-align: middle;
        margin-right: 10px;
    }
    </style>

    <div class="footer">
        <img src="https://r3group-project.com/wp-content/uploads/2023/05/European_Union_flag_with_blue_background_and_yellow_stars_generated.jpg" alt="EU Logo">
        The R3GROUP project has received funding from the European Union’s Horizon Europe programme under grant agreement No. 101091869.
    </div>
 """,
 unsafe_allow_html=True
)



