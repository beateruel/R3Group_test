import streamlit as st
import json
import subprocess
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from lib import GLN_IO as gl
import GLN_optimization as gopt
import GLN_simulation_1 as gsimu

def check_serializability(obj):
    try:
        json.dumps(obj)
        return True
    except TypeError:
        return False

# --- Página principal con tabs ---
df_products_material = gl.read_products_BOM()
batch_size_init = int(df_products_material["BatchSize"].values[0])
df_materials_suppliers = gl.read_material_suppliers()
df_supplier_material = df_materials_suppliers[(df_materials_suppliers["SupplierId"]=="1") & (df_materials_suppliers["MaterialId"]=="1")]
reorder_quantity_init = int(df_supplier_material["ReorderQuantity"].iloc[0])   
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
    if "show_help" not in st.session_state:
        st.session_state.show_help = False

    if st.button("❓"):
        st.session_state.show_help = not st.session_state.show_help

    if st.session_state.show_help:
        st.info("Select .csv files with the crude oil prices and the PBT prices. Check the manual for the structure.")
    
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
    st.session_state.simulation_as_is_done = False
    st.session_state.simulation_to_be_done = False
    st.header("Scenario simulation")     

    # Row 1: Input + Graphs
    top_col1, top_col2, top_col3 = st.columns([1, 2, 2])

    # Layout 1: Inputs
    
    param1="AS_IS"
    param2=0
    param3=-1
    param4=-1
    params_as_is = {
                "SimulationName": param1,
                "DemandVariation": param2,
                "InitWeekShortage": param3,
                "EndWeekShortage": param4,
                "BatchSize":batch_size_init,
                "ReorderQuantity":reorder_quantity_init
            }

    param1 ="STRESS_TEST"
    with top_col1:
        st.subheader("Input parameters: STRESS TEST")       
        param2 = st.number_input("Demand variation (-1.0, 1.0)", value=0.0)
        param3 = st.number_input("Week starting the supply shortage", value=-1)
        param4 = st.number_input("Week ending the supply shortage", value=-1)
    params_to_be = {
                "SimulationName": param1,
                "DemandVariation": param2,
                "InitWeekShortage": param3,
                "EndWeekShortage": param4, 
                "BatchSize":batch_size_init,
                "ReorderQuantity":reorder_quantity_init
        }
    print("\antes de entrar")
    if st.button("Start simulation"):
        input_file = os.path.join(BASE_DIR, "input", "main_parameters.json")
        
        with open(input_file, "w") as f:
            json.dump(params_as_is, f)
            # Leer y validar
        with open(input_file, "r") as f:
            data = json.load(f)
            print("Contenido leído as is:", data)

        # result_as_is = subprocess.run(
        #     [sys.executable, os.path.join(BASE_DIR, "GLN_simulation1.2.py")],
        #     capture_output=True,
        #     text=True
        #     )
        df_as_is=gsimu.main()
        st.session_state.simulation_as_is_done = True
        # st.session_state.simulation_output = result_as_is.stdout   
        
        input_file = os.path.join(BASE_DIR, "input", "main_parameters.json")
        with open(input_file, "w") as f:            
            json.dump(params_to_be, f)
        with open(input_file, "r") as f:
            data = json.load(f)
            print("\nContenido leído to be:", data)
        # result_to_be = subprocess.run(
        #     [sys.executable, os.path.join(BASE_DIR, "GLN_simulation1.2.py")],
        #     capture_output=True,
        #     text=True
        #     )
        df_to_be=gsimu.main()
        st.session_state.simulation_as_is_done = True
        st.session_state.simulation_to_be_done = True
        
        gl.write_interface_results(df_to_be, "AS_IS")

    if st.button("Clean"):
        st.session_state.simulation_to_be_done = False
        st.session_state.simulation_as_is_done = False
        st.rerun()

    # Layout 2: Graphs
    
    if  st.session_state.simulation_as_is_done==True and st.session_state.simulation_to_be_done==True:
        st.success("Simulation completed")
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
            if os.path.exists(output_file_2):
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
  
    st.header("Scenario simulation optimization")
    st.session_state.simulation_conventional_stress_test_done = False
    st.session_state.simulation_optimal_stress_test_done = False
        

    # Row 1: Input + Graphs
    top_col1, top_col2, top_col3 = st.columns([1, 2, 2])

    # Layout 1: Inputs
    with top_col1:
        st.subheader("Input parameters: OPTIMAL STRESS TEST")  
        param1="CONVENTIONAL_STRESS_TEST"
        param2=st.number_input("Demand variation optimal (-1.0, 1.0)", value=0.0)
        param3=-1
        param4=-1
        params_conventional = {
                    "SimulationName": param1,
                    "DemandVariation": param2,
                    "InitWeekShortage": param3,
                    "EndWeekShortage": param4,
                    "BatchSize":batch_size_init,
                    "ReorderQuantity":reorder_quantity_init
                }

        param1 ="OPTIMAL_STRESS_TEST"
        EBQ=gopt.EBQ(param2)
        print(f"\n EBQ {EBQ}")
        EOQ=gopt.EOQ(param2)
        print(f"\n EOQ {EOQ}")
   
            
       # param2 = st.number_input("Demand variation for optimization (-1.0, 1.0)", value=0.0)
        params_optimal = {
                "SimulationName": param1,
                "DemandVariation": param2,
                "InitWeekShortage": param3,
                "EndWeekShortage": param4, 
                "BatchSize":EBQ,
                "ReorderQuantity":EOQ
        }
    
    if st.button("Start simulation optimization"):
        input_file = os.path.join(BASE_DIR, "input", "main_parameters.json")
        
        with open(input_file, "w") as f:
            json.dump(params_conventional, f)
            # Leer y validar
        with open(input_file, "r") as f:
            data = json.load(f)
            print("Contenido conventional:", data)

        df_conventional=gsimu.main()
        name='output/simulation'
        simulation_data=gl.write_output_json(df_conventional,name + '.json')
        st.session_state.simulation_conventional_stress_test_done = True
         
        
        input_file = os.path.join(BASE_DIR, "input", "main_parameters.json")
        with open(input_file, "w") as f:            
            json.dump(params_optimal, f)
        with open(input_file, "r") as f:
            data = json.load(f)
            print("\nContenido optimal:", data)
     
        df_optimal=gsimu.main()       
        st.session_state.simulation_optimal_stress_test_done = True
        
        gl.write_interface_results(df_optimal, "CONVENTIONAL_STRESS_TEST")

    if st.button("Clean optimization"):
        st.session_state.simulation_conventional_stress_test_done = False
        st.session_state.simulation_optimal_stress_test_done = False
        st.rerun()
    
    
     
    with top_col2:        
        st.write(f"📦 Inital Batch size quantity: **{batch_size_init}** pieces")
        st.write(f"📦OPtimal Batch size quantity: **{EBQ}** kg PBT")
        
           
    with top_col3:        
        st.write(f"📦Inital Minimum order quantity: **{reorder_quantity_init}** pieces")
        st.write(f"📦OPtimal Minimum order quantity: **{EOQ}** kg PBT")
   
  # # Layout 2: Graphs
    
    if  st.session_state.simulation_optimal_stress_test_done==True:
        st.success("Simulation optimization completed")
       
        with top_col2:
            st.subheader("CONVENTIONAL STRESS TEST")
            output_file_1 = os.path.join(BASE_DIR, "output", "CONVENTIONAL_STRESS_TEST_individual.png")
            if os.path.exists(output_file_1):
                st.image(output_file_1, caption="CONVENTIONAL STRESS TEST")
            output_file_2 = os.path.join(BASE_DIR, "output", "CONVENTIONAL_STRESS_TEST_combined.png")
            if os.path.exists(output_file_1):
                st.image(output_file_2, caption="CONVENTIONAL STRESS TEST")

        with top_col3:
            st.subheader("OPTIMAL STRESS TEST")
            output_file_1 = os.path.join(BASE_DIR, "output", "OPTIMAL_STRESS_TEST_individual.png")
            if os.path.exists( output_file_1):
                st.image(output_file_1, caption=param1)
            output_file_2 = os.path.join(BASE_DIR, "output", "OPTIMAL_STRESS_TEST_combined.png")
            if os.path.exists(output_file_2):
                st.image(output_file_2, caption=param1)

    # Row 2: KPI Table (full width)st.markdown("---")


        st.markdown("---")
        st.subheader("Simulation optimization - Scenarios comparison")

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



