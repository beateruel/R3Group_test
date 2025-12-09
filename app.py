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
import pickle

def interpret_mape(mape_in):
    
    
    if isinstance(mape_in, set) and len(mape_in) == 1:
        mape_in = next(iter(mape_in))  # Extrae el único valor
        print(type(mape_in))
        mape=mape_in*100
    else:
        exit(0)  
    
    if mape < 10:
        st.write(f"MAPE: {mape:.2f}% presents an excellent accuracy as the error is <10%")
    elif 10 <= mape < 20:
        st.write(f"MAPE: {mape:.2f}% presents a good accuracy as the error is between 10% and 20%")
    elif 20 <= mape < 50:
        st.write(f"MAPE: {mape:.2f}% presents an acceptable accuracy as the error is between 20% and 50%")
    else:
        st.write(f"MAPE: {mape:.2f}% presents a poor accuracy as the error is >50%")



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
    st.markdown("""
    Welcome to the **supply chain digital twin tool**.

    This application allows you to:
    - Enter parameters
    - Launch simulations
    - Compare results between scenarios
    """)
    st.divider()

# --- Prediction Page ---
with tabs[1]:
    st.title("Price Prediction")

    # 🔹 Input Section
    st.subheader("Input parameters")
    input_col, help_col = st.columns([2, 1])

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with input_col:
        raw_material_file = st.file_uploader(
            "Upload raw material prices (CSV)",
            type="csv",
            help="CSV file with crude oil historical prices. Columns: Date, Price."
        )
        main_material_file = st.file_uploader(
            "Upload supplier material prices (CSV)",
            type="csv",
            help="CSV file with supplier-specific supplier material prices. Columns: Date, Price."
        )

        if st.button("Start prediction"):
            # Persistimos en session_state
            st.session_state['run_id'] = __import__('time').strftime("%Y%m%d_%H%M%S")
            st.session_state['raw_path'] = None
            st.session_state['main_path'] = None

            # Guardar los CSV en BASE_DIR
            if raw_material_file:
                raw_path = os.path.join(BASE_DIR, raw_material_file.name)
                with open(raw_path, "wb") as f:
                    f.write(raw_material_file.getbuffer())
                st.session_state['raw_path'] = raw_path
                st.success(f"Raw material prices saved to {raw_path}")

            if main_material_file:
                main_path = os.path.join(BASE_DIR, main_material_file.name)
                with open(main_path, "wb") as f:
                    f.write(main_material_file.getbuffer())
                st.session_state['main_path'] = main_path
                st.success(f"Supplier material prices saved to {main_path}")

            # Ejecutar el generador SOLO si tenemos ambos CSV
            if st.session_state['raw_path'] and st.session_state['main_path']:
                # Registrar tiempos de modificación previos (si existen)
                raw_png = os.path.join(OUTPUT_DIR, "raw_material_prices.png")
                pred_png = os.path.join(OUTPUT_DIR, "predictions.png")
                prev_raw_mtime = os.path.getmtime(raw_png) if os.path.exists(raw_png) else 0
                prev_pred_mtime = os.path.getmtime(pred_png) if os.path.exists(pred_png) else 0

                try:
                    result = subprocess.run(
                        [
                            sys.executable,
                            os.path.join(BASE_DIR, "GRAL_forecasting_v1.1.py"),
                            st.session_state['raw_path'],
                            st.session_state['main_path']
                        ],
                        cwd=BASE_DIR,            # 🔒 asegura rutas relativas del script
                        capture_output=True,
                        text=True,
                        check=True               # 🔒 lanza excepción si returncode != 0
                    )
                    st.info("Prediction process launched. See results below.")
                    st.session_state['stdout'] = result.stdout
                    st.session_state['stderr'] = result.stderr

                    # Validar que los PNG se han actualizado
                    ok_raw = os.path.exists(raw_png) and os.path.getmtime(raw_png) > prev_raw_mtime
                    ok_pred = os.path.exists(pred_png) and os.path.getmtime(pred_png) > prev_pred_mtime

                    if not (ok_raw and ok_pred):
                        st.warning("The generator ran but output images did not update. Check logs below.")

                except subprocess.CalledProcessError as e:
                    st.error("Prediction script failed.")
                    st.code(e.stdout or "", language="text")
                    st.code(e.stderr or "", language="text")

    with help_col:
        with st.expander("ℹ️ What files should I upload?"):
            st.markdown("""
- **Raw material prices** → CSV with crude oil time series.
- **Supplier material prices** → CSV with supplier-specific material historical prices.

The model uses **XGBoost** for forecasting raw material prices (Stage 1),
then predicts **supplier material prices** from the last 12 forecasted values (Stage 2).
            """)

    st.divider()

    # 🔹 Results Section
    st.subheader("Forecast Results")
    result_tabs = st.tabs(["📈 Raw material forecast", "📉 Supplier material price forecast"])

    def show_image_bytes(path: str, caption: str):
        if os.path.exists(path):
            with open(path, "rb") as f:
                st.image(f.read(), caption=caption)
        else:
            st.info(f"Not found: {path}")

    with result_tabs[0]:
        raw_png = os.path.join(OUTPUT_DIR, "raw_material_prices.png")
        show_image_bytes(raw_png, "Historical and forecasted raw material prices")

        # scores.pickle robusto
        pkl_path = os.path.join(OUTPUT_DIR, 'scores.pickle')
        if os.path.exists(pkl_path):
            try:
                with open(pkl_path, 'rb') as handle:
                    resultsDict = pickle.load(handle)
                # ⚠️ usar una única clave consistente
                mape_key = 'XGBoost' if 'XGBoost' in resultsDict else ('xgb' if 'xgb' in resultsDict else None)
                if mape_key:
                    mape_error = {resultsDict[mape_key]['mape']}
                    interpret_mape(mape_error)
                else:
                    st.warning("MAPE key not found in scores.pickle")
            except Exception as ex:
                st.warning(f"Could not read scores.pickle: {ex}")

        with st.expander("ℹ️ Explanation"):
            st.markdown("""
This graph shows the last 12 **raw material price forecast** generated using XGBoost.
They will be used to project supplier material prices.
            """)

    with result_tabs[1]:
        pred_png = os.path.join(OUTPUT_DIR, "predictions.png")
        show_image_bytes(pred_png, "Predicted supplier material prices vs raw material forecast")

        # Reutilizamos el mismo pkl y clave consistente
        pkl_path = os.path.join(OUTPUT_DIR, 'scores.pickle')
        if os.path.exists(pkl_path):
            try:
                with open(pkl_path, 'rb') as handle:
                    resultsDict = pickle.load(handle)
                mape_key = 'XGBoost' if 'XGBoost' in resultsDict else ('xgb' if 'xgb' in resultsDict else None)
                if mape_key:
                    mape_error = {resultsDict[mape_key]['mape']}
                    interpret_mape(mape_error)
                else:
                    st.warning("MAPE key not found in scores.pickle")
            except Exception as ex:
                st.warning(f"Could not read scores.pickle: {ex}")

        with st.expander("ℹ️ Explanation"):
            st.markdown("""
This graph compares **supplier-specific material prices** against the **predicted supplier material prices**.
The forecasted values are derived via simple linear regression from the last 12 XGBoost raw material forecasts.
            """)

    # Mostrar logs si existen
    #if 'stdout' in st.session_state or 'stderr' in st.session_state:
        #st.divider()
        #st.subheader("Generator logs")
        #st.code(st.session_state.get('stdout', ''), language="text")
        #st.code(st.session_state.get('stderr', ''), language="text")


# --- Simulation Page ---
with tabs[2]:
    st.header("Scenario Simulation")

    # 🔹 Generic function for KPI charts
    def plot_kpi_distribution(df_kpis, title):
        """Show KPI chart:
           - Pie chart if all values >= 0
           - Bar chart if there are negative values
        """
        fig, ax = plt.subplots()

        if (df_kpis["Value"] < 0).any():
            # At least one negative → use bar chart
            colors = ["green" if v >= 0 else "red" for v in df_kpis["Value"]]
            ax.bar(df_kpis["KPI"], df_kpis["Value"], color=colors)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_ylabel("Value")
            ax.set_title(f"{title} (bar chart - negatives detected)")
            plt.xticks(rotation=45, ha="right")
        else:
            # All values >= 0 → use pie chart
            ax.pie(df_kpis["Value"], labels=df_kpis["KPI"], autopct='%1.1f%%')
            ax.set_title(f"{title} (distribution)")

        return fig

    # 🔹 Input layout
    input_col, help_col = st.columns([2, 1])

    with input_col:
        st.subheader("Input parameters: Stress Test")

        demand_var = st.number_input(
            "Demand variation (-1.0 to 1.0)", 
            value=0.0, step=0.1,
            help="Percentage change in demand (e.g., 0.2 = +20%)."
        )

        start_week = st.number_input(
            "Week starting the supply shortage", 
            value=-1, step=1,
            help="Week when supply interruption begins. -1 = no interruption."
        )

        end_week = st.number_input(
            "Week ending the supply shortage", 
            value=-1, step=1,
            help="Week when supply interruption ends. -1 = no interruption."
        )

        if st.button("🚀 Start simulation"):
            params_as_is = {
                "SimulationName": "AS_IS",
                "DemandVariation": 0,
                "InitWeekShortage": -1,
                "EndWeekShortage": -1,
                "BatchSize": batch_size_init,
                "ReorderQuantity": reorder_quantity_init
            }

            input_file = os.path.join(BASE_DIR, "input", "main_parameters.json")
        
            with open(input_file, "w") as f:
                json.dump(params_as_is, f)
                # Leer y validar
            with open(input_file, "r") as f:
                data = json.load(f)
                print("Contenido AS IS:", data)
            df_as_is = gsimu.main()             
            name='output/simulation'
            simulation_data=gl.write_output_json(df_as_is,name + '.json')
            st.session_state.simulation_as_is_done = True

            params_to_be = {
                "SimulationName": "STRESS_TEST",
                "DemandVariation": demand_var,
                "InitWeekShortage": start_week,
                "EndWeekShortage": end_week,
                "BatchSize": batch_size_init,
                "ReorderQuantity": reorder_quantity_init
            }

            with open(input_file, "w") as f:
                json.dump(params_to_be, f)
                # Leer y validar
            with open(input_file, "r") as f:
                data = json.load(f)
                print("Contenido AS IS:", data)
            df_to_be = gsimu.main()
            st.session_state.simulation_to_be_done = True
            simulation_data=gl.write_output_json(df_to_be,name + '.json')
            gl.compare_simulations(name + '.json')

    with help_col:
        with st.expander("ℹ️ About the input parameters"):
            st.markdown("""
            - **Demand variation**: percentage change in total product demand.  
              Example: `0.2 = +20%`, `-0.3 = -30%`.  
            - **Supply shortage (start & end week)**: period when **no material is delivered by supplier**.  
              If both values are `-1`, **no shortage is applied**.
            """)

    st.divider()

    # 🔹 Comparative graphs
    if st.session_state.get("simulation_as_is_done") and st.session_state.get("simulation_to_be_done"):
        st.success("Simulation completed ✅")

        graph_col1, graph_col2 = st.columns(2)

        with graph_col1:
            st.subheader("AS-IS Scenario")
            ind_file = os.path.join(BASE_DIR, "output", "AS_IS_individual.png")
            comb_file = os.path.join(BASE_DIR, "output", "AS_IS_combined.png")
            if os.path.exists(ind_file):
                st.image(ind_file, caption="Inventory & demand evolution (AS-IS)")
            if os.path.exists(comb_file):
                st.image(comb_file, caption="Inventory on-had  vs in-transit (AS-IS)")

        with graph_col2:
            st.subheader("Stress Test Scenario")
            ind_file = os.path.join(BASE_DIR, "output", "STRESS_TEST_individual.png")
            comb_file = os.path.join(BASE_DIR, "output", "STRESS_TEST_combined.png")
            if os.path.exists(ind_file):
                st.image(ind_file, caption="Inventory & demand evolution (Stress Test)")
            if os.path.exists(comb_file):
                st.image(comb_file, caption="Inventory on-hand vs in-transit (Stress Test)")

        st.divider()

        # 🔹 KPIs
        st.subheader("Scenario KPI Comparison")

        output_file = os.path.join(BASE_DIR, "output", "comparison.json")
        if os.path.exists(output_file):
            with open(output_file) as f:
                df = json.load(f)

            last_scenario = list(df.keys())[-1]
            items = list(df[last_scenario].items())

            financial_kpis = pd.DataFrame(items[3:6], columns=["KPI", "Value"])
            operational_kpis = pd.DataFrame(items[-7:], columns=["KPI", "Value"])

            kpi_tabs = st.tabs(["💰 Financial KPIs", "⚙️ Operational KPIs"])

            # --- Financial KPIs
            with kpi_tabs[0]:
                
                st.markdown("**AS-IS vs Stress Test (Bar chart)**")
                fig, ax = plt.subplots()
                ax.bar(financial_kpis["KPI"], financial_kpis["Value"])
                ax.set_ylabel("Value")
                ax.set_title("Financial KPIs comparison")               
                st.pyplot(fig)

            # --- Operational KPIs
            with kpi_tabs[1]:
               
                st.markdown("**AS-IS vs Stress Test (Bar chart)**")
                fig, ax = plt.subplots()
                ax.bar(operational_kpis["KPI"], operational_kpis["Value"])
                ax.set_ylabel("Value")
                ax.set_title("Operational KPIs comparison")
                plt.xticks(rotation=45, ha="right")
                st.pyplot(fig)

            with st.expander("ℹ️ KPI explanation"):
                st.markdown("""
                - **Financial KPIs**: costs, margins, and economic efficiency.  
                - **Operational KPIs**: inventory levels, lead time, demand fulfillment, etc.  

                👉 Pie chart is shown only if all values are positive.  
                👉 If there are negatives, a bar chart is used (green = positive, red = negative).  
                👉 The bar chart directly compares **AS-IS vs Stress Test**.
                """)


        # --- Optimization  Page ---
with tabs[3]:
  
    st.header("Scenario Simulation Optimization")    
     
    st.session_state.simulation_conventional_stress_test_done = False
    st.session_state.simulation_optimal_stress_test_done = False

    # 🔹 Input layout
    input_col, value_col  = st.columns([2, 2])

    # Layout 1: Inputs
    with input_col:
        st.subheader("Input parameters: Optimal Stress Test")

        demand_var_opt = st.number_input(
            "Demand variation (-1.0 to 1.0)", 
            value=0.0, step=0.1,
            help="Percentage change in demand for optimization."
        )

        # Build parameter dictionaries
        params_conventional = {
            "SimulationName": "CONVENTIONAL_STRESS_TEST",
            "DemandVariation": demand_var_opt,
            "InitWeekShortage": -1,
            "EndWeekShortage": -1,
            "BatchSize": batch_size_init,
            "ReorderQuantity": reorder_quantity_init
        }

        EBQ = gopt.EBQ(demand_var_opt)
        EOQ = gopt.EOQ(demand_var_opt)

        params_optimal = {
            "SimulationName": "OPTIMAL_STRESS_TEST",
            "DemandVariation": demand_var_opt,
            "InitWeekShortage": -1,
            "EndWeekShortage": -1,
            "BatchSize": EBQ,
            "ReorderQuantity": EOQ
        }
    
    if st.button("🚀 Start optimization simulation"):
        # Conventional run
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
         
        # Optimal run
        input_file = os.path.join(BASE_DIR, "input", "main_parameters.json")
        with open(input_file, "w") as f:            
            json.dump(params_optimal, f)
        with open(input_file, "r") as f:
            data = json.load(f)
            print("\nContenido optimal:", data)
     
        df_optimal=gsimu.main()       
        st.session_state.simulation_optimal_stress_test_done = True        
        simulation_data=gl.write_output_json(df_optimal,name + '.json')
        gl.compare_simulations(name + '.json')

    if st.button("🧹 Reset optimization"):
        st.session_state.simulation_conventional_stress_test_done = False
        st.session_state.simulation_optimal_stress_test_done = False
        st.rerun()
    
    with value_col:
        st.subheader("Batch & Order Quantities")
        st.write(f"📦 **Initial Batch Size**: {batch_size_init} units")
        st.write(f"📦 **Optimized Batch Size (EBQ)**: {EBQ} kg supplier material")
        st.write(f"📦 **Initial Reorder Quantity**: {reorder_quantity_init} units")
        st.write(f"📦 **Optimized Reorder Quantity (EOQ)**: {EOQ} kg supplier material")

        with st.expander("ℹ️ Explanation of EBQ & EOQ"):
            st.markdown("""
            - **EBQ (Economic Batch Quantity)**: the optimal production lot size,  
              recalculated according to demand variation.  
            - **EOQ (Economic Order Quantity)**: the optimal minimum supplier order quantity.  

            These values are updated dynamically as the demand variation changes.
            """)


    st.divider()
   
  # 🔹 Show optimization results
    if st.session_state.get("simulation_optimal_stress_test_done"):
        st.success("Optimization simulation completed ✅")

        graph_col1, graph_col2 = st.columns(2)

        with graph_col1:
            st.subheader("Conventional Stress Test")
            ind_file = os.path.join(BASE_DIR, "output", "CONVENTIONAL_STRESS_TEST_individual.png")
            comb_file = os.path.join(BASE_DIR, "output", "CONVENTIONAL_STRESS_TEST_combined.png")
            if os.path.exists(ind_file):
                st.image(ind_file, caption="Conventional stress test (individual)")
            if os.path.exists(comb_file):
                st.image(comb_file, caption="Conventional stress test (combined)")

        with graph_col2:
            st.subheader("Optimal Stress Test")
            ind_file = os.path.join(BASE_DIR, "output", "OPTIMAL_STRESS_TEST_individual.png")
            comb_file = os.path.join(BASE_DIR, "output", "OPTIMAL_STRESS_TEST_combined.png")
            if os.path.exists(ind_file):
                st.image(ind_file, caption="Optimal stress test (individual)")
            if os.path.exists(comb_file):
                st.image(comb_file, caption="Optimal stress test (combined)")

        st.divider()
        
        # 🔹 KPIs
        st.subheader("Scenario KPI Comparison")

        output_file = os.path.join(BASE_DIR, "output", "comparison.json")
        if os.path.exists(output_file):
            with open(output_file) as f:
                df = json.load(f)

            last_scenario = list(df.keys())[-1]
            items = list(df[last_scenario].items())

            financial_kpis = pd.DataFrame(items[3:6], columns=["KPI", "Value"])
            operational_kpis = pd.DataFrame(items[-7:], columns=["KPI", "Value"])

            kpi_tabs = st.tabs(["💰 Financial KPIs", "⚙️ Operational KPIs"])

            # --- Financial KPIs
            with kpi_tabs[0]:
                
                st.markdown("**Conventional Stress Test vs Optimal Stress Test (Bar chart)**")
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(financial_kpis["KPI"], financial_kpis["Value"])
                ax.set_ylabel("Value")
                ax.set_title("Financial KPIs comparison")
                st.pyplot(fig)

            # --- Operational KPIs
            with kpi_tabs[1]:
               
                st.markdown("**Conventional Stress Test vs Optimal Stress Test (Bar chart)**")
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(operational_kpis["KPI"], operational_kpis["Value"])
                ax.set_ylabel("Value")
                ax.set_title("Operational KPIs comparison")
                plt.xticks(rotation=45, ha="right")
                st.pyplot(fig)

            with st.expander("ℹ️ KPI explanation"):
                st.markdown("""
                - **Financial KPIs**: costs, margins, and economic efficiency.  
                - **Operational KPIs**: inventory levels, lead time, demand fulfillment, etc.  

                👉 Pie chart is shown only if all values are positive.  
                👉 If there are negatives, a bar chart is used (green = positive, red = negative).  
                👉 The bar chart directly compares **Conventional Stress Test vs Optimal Stress Test (Bar chart)**.
                """)

                   
    
         
    


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



