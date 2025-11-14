import pandas as pd
import json
import pdb
import numpy as np
import os


#----------------------------------------------------------------------------------------------------------------------------
#extract the monthly orders from orders.json and create the DataFrame with monhtly orders 
def read_monhtly_demand(variation):
    #variation: decimal between -1 and1 with the percentage of demand variation
    
    demand_week =pd.DataFrame(columns=['Month', 'ProductId', 'ProductName','Quantity'])
    
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, "..", "input", "demand2.json") 
    df_demand_month_in=pd.read_json(file_path, orient="index")
    df_demand_month=pd.DataFrame()
   
    
    for dm in df_demand_month_in:
        month=df_demand_month_in[dm].iloc[0]['month']
        items=df_demand_month_in[dm].iloc[0]['Items']
         
        for i in items:
           
            values = {'Month': month,'ProductId':i['ProductId'], 'ProductName': i['ProductName'], 'Quantity': int(i['Quantity']*(1+variation))}          
            columns = ['Month', 'ProductId', 'ProductName', 'Quantity']
            index=['0']
            
            df_2 = pd.DataFrame(data=values, columns=columns, index=index)
            df_demand_month = pd.concat([df_demand_month, df_2],ignore_index=True)      
            
            # Build the relative path to the output file
            output_path = os.path.join(base_path, "..", "output", "input_monthly_demand.csv")

            df_demand_month.to_csv(output_path)
            

    
    return df_demand_month

#----------------------------------------------------------------------------------------------------------------------------
#extract the monthly orders from orders.json and create the DataFrame with weekly orders whose demand is uniformly distributed
def read_weekly_demand(variation):
     #variation: decimal between -1 and1 with the percentage of demand variation
    demand_week =pd.DataFrame(columns=['Month', 'Week', 'ProductId', 'ProductName','Quantity'])
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, "..", "input", "demand2.json") 
    df_demand_month=pd.read_json(file_path, orient="index")

   
    df_demand_week=pd.DataFrame()
    weeks_month=range(4)
    week=0

    for dm in df_demand_month:
        month=df_demand_month[dm].iloc[0]['month']
        items=df_demand_month[dm].iloc[0]['Items']
        for w in weeks_month:           
            for i in items:
                values = {'Month': month, 'Week':week,'ProductId':i['ProductId'], 'ProductName': i['ProductName'], 'Quantity': int((i['Quantity']*(1+variation))/4)}          
                columns = ['Month', 'Week', 'ProductId', 'ProductName', 'Quantity']
                index=['0']
                df_2 = pd.DataFrame(data=values, columns=columns, index=index)
                df_demand_week = pd.concat([df_demand_week, df_2],ignore_index=True)                               
            week+=1
    
    #print(df_demand_week.head(10))
    # Build the relative path to the output file
    output_path = os.path.join(base_path,"..", "output", "input_weekly_demand.csv")
    df_demand_week.to_csv(output_path)
    return df_demand_week

#----------------------------------------------------------------------------------------------------------------------------
#read the characteristics of the products the materials (level 1) annd suppliers from BOM.json and retruns a dataframe with 
#this data

def read_products_BOM():
    df_products_BOM=pd.DataFrame() 
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, "..", "input", "BOM.json")                
    df_products_BOM_aux=pd.read_json(file_path, orient="index")
    
    for pb in df_products_BOM_aux:
        element =df_products_BOM_aux[pb].iloc[0]
        materials=element['Materials']
        for material in materials:
            # suppliers=material['Suppliers']
            # for supplier in suppliers:
            values= { 'ProductId': element['ProductId'],
                            'ProductName':element['ProductName'],
                            'WeightUnitOfMeasure':element['WeightUnitOfMeasure'],
                            'TotalWeight':element['TotalWeight'], 
                            'CycleTime':element['CycleTime'],
                            'ProductionRate':element['ProductionRate'], 
                            'ScrapPercentage':element['ScrapPercentage'],
                            'MaterialId':material['MaterialId'],
                            'MaterialName':material['MaterialName'],
                            'MaterialDisplayName':material['MaterialDisplayName'],
                            'MaterialWeightUnitOfMeasure':material['MaterialWeightUnitOfMeasure'],
                            'MaterialTotalWeight':material['MaterialTotalWeight'],
                            'MaterialScrapPercentage':material['MaterialScrapPercentage'],
                            'MaterialOilPercentage':material['MaterialOilPercentage'],
                            'SetupTime': element['SetupTime'],
                            'SetupCost': element['SetupCost'],
                            'ProductionCost':element['ProductionCost'],             
                            'QualityCheckTime':element['QualityCheckTime'],
                            'BatchSize':element['BatchSize'],
                            'SafetyStock':element['SafetyStock']}
            columns = ['ProductId', 'ProductName', 'WeightUnitOfMeasure','TotalWeight','CycleTime','ProductionRate', 'ScrapPercentage','MaterialId','MaterialName','MaterialDisplayName','MaterialWeightUnitOfMeasure','MaterialTotalWeight','MaterialScrapPercentage','MaterialOilPercentage', 'SetupTime','SetupCost','ProductionCost', 'QualityCheckTime', 'BatchSize', 'SafetyStock']
            index=['0']
            df_2 = pd.DataFrame(data=values, columns=columns, index=index)
            df_products_BOM = pd.concat([df_products_BOM, df_2],ignore_index=True)             
   
    #print(df_products_BOM)
    return (df_products_BOM)
 
    
#-----------------------------------------------------------------------------------------------------------
#extract the supplier per material

def read_material_suppliers():

    df_material_suppliers=pd.DataFrame()
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, "..", "input", "suppliers.json")        
    df_material_suppliers_aux=pd.read_json(file_path, orient="index")
 
    for pb in df_material_suppliers_aux:
      
        supplier =df_material_suppliers_aux[pb].iloc[0]
        values={'SupplierId':supplier['SupplierId'],
                            'MaterialId':supplier['MaterialId'],
                            'MaterialDisplayName':supplier['MaterialDisplayName'],
                            'SupplierName': supplier['SupplierName'],
                            'SupplierLocation': supplier['SupplierLocation'],
                            'SupplierDistanceUnitOfMeasure': supplier['SupplierDistanceUnitOfMeasure'],
                            'SupplierDistance': supplier['SupplierDistance'],
                            'SupplierPriceUnitOfMeasure': supplier['SupplierPriceUnitOfMeasure'],
                            'SupplierMinPrice': supplier['SupplierMinPrice'],
                            'SupplierMaxPrice': supplier['SupplierMaxPrice'],
                            'ReorderWeightUnitOfMeasure': supplier['ReorderWeightUnitOfMeasure'],
                            'ReorderQuantity': supplier['ReorderQuantity'],
                            'SupplierTimeUnitOfMeasure': supplier['SupplierTimeUnitOfMeasure'],
                            'ReorderPoint': supplier['ReorderPoint'],
                            'SupplierLeadTime': supplier['SupplierLeadTime'],
                            'SupplierDeliveryQuantity': supplier['SupplierDeliveryQuantity'],
                            'SupplierDeliveryTime': supplier['SupplierDeliveryTime'],
                            'SupplierDeliverMode': supplier['SupplierDeliverMode'],
                            'SupplierSafetyStock': supplier['SupplierSafetyStock']}
        columns = ['SupplierId','MaterialId', 'MaterialDisplayName', 'SupplierName','SupplierLocation','SupplierDistanceUnitOfMeasure','SupplierDistance', 'SupplierPriceUnitOfMeasure','SupplierMinPrice','SupplierMaxPrice','ReorderWeightUnitOfMeasure','ReorderQuantity','SupplierTimeUnitOfMeasure','ReorderPoint',
'SupplierLeadTime','SupplierDeliveryQuantity','SupplierDeliveryTime','SupplierDeliverMode','SupplierSafetyStock']
        index=['0']
        df_2 = pd.DataFrame(data=values, columns=columns, index=index)
        df_material_suppliers = pd.concat([df_material_suppliers, df_2],ignore_index=True)      
                
        #print(df_material_suppliers['ReorderQuantity'].values)    
        return (df_material_suppliers)

#-----------------------------------------------------------------------------------------------------------
#extract the supplier per material

def read_predicted_supplier_material_prices():

    PBT_price=np.zeros(15)
    #file = "C:/Users/broyo/R3Group_env/GLN Forecasting/output/forecast_GLN.csv"
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, "..", "input", "forecast_GLN.csv")   
    #file="forecast_GLN.csv"
    
    df_PBT_prices=pd.read_csv(file_path, delimiter=',', header=None)
    df_PBT_prices= df_PBT_prices.tail(15)
      
    for month in range(0,15):
        #for price in df_PBT_prices.tail(15):
        price = df_PBT_prices.iloc[[month],[2]].iloc[0]       
        PBT_price[month] =round(float(price), 2)
        
   # print(PBT_price)         
    
    return (PBT_price)
#-----------------------------------------------------------------------------------------------------------
#extract the parameters to configure diferent scenarios

def read_main_parameters():

    df_main_parameters=pd.DataFrame() 
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, "..", "input", "main_parameters.json")           
    df_main_parameters=pd.read_json(file_path, orient="index")
    
          
    return (df_main_parameters)


#-----------------------------------------------------------------------------------------------------------

#extract the parameters to configure diferent scenarios

def write_output_json(new_data,file_path):
    """
    Appends a DataFrame to an existing JSON file. 
    If the file doesn't exist, it creates a new one.

    Parameters:
    - file_path (str): Path to the JSON file.
    - new_data (pd.DataFrame): The DataFrame to append.

    Returns:
    - None
    """
    try:
        # Step 1: Load existing JSON data
        with open(file_path, 'r') as f:
            existing_data = json.load(f)

        # Check if the existing data is a list or a dictionary
        if isinstance(existing_data, dict):
            # If it's a dictionary, we may want to convert it into a list of values or handle it differently
            existing_data = [existing_data]  # Convert it into a list of dictionaries
        
    except FileNotFoundError:
        # If the file doesn't exist, start with an empty list
        existing_data = []

    # Step 2: Convert new DataFrame to JSON records format (list of dictionaries)
    new_data_json = new_data.to_dict(orient='records')

    # Step 3: Append the new data
    existing_data.extend(new_data_json)

    # Step 4: Write the updated data back to the JSON file
    with open(file_path, 'w') as f:
        json.dump(existing_data, f, indent=4)
        
    return existing_data
    #print("Data successfully appended to the JSON file!")
   

#-----------------------------------------------------------------------------------------------------------
   
def write_output_csv( new_data, file_path):
    """
    Appends a DataFrame to an existing CSV file.
    Creates the file if it does not exist.

    Parameters:
    - file_path (str): Path to the CSV file.
    - new_data (pd.DataFrame): The DataFrame to append.

    Returns:
    - None
    """
    # Check if file exists
    file_exists = os.path.isfile(file_path)

    # Write or append to the CSV file
    new_data.to_csv(file_path, mode='a', header=not file_exists, index=False)

    #print(f"Data successfully written to {file_path}!")


#-----------------------------------------------------------------------------------------------------------------------
def compare_simulations_old(simulation_data, base_simulation_name):
        # Create a dictionary to store the results
        results = {}
        #print(f" simulation data {simulation_data}")
        # Extract the base simulation data
        base_simulation = None
        for scenario in simulation_data:
            if scenario['simulation_name'] == base_simulation_name:
                base_simulation = scenario
                break
        
        if base_simulation is None:
            raise ValueError(f"Base simulation '{base_simulation_name}' not found in the data.")
        
        # Compare the base simulation with the other scenarios
        for scenario in simulation_data:
            if scenario['simulation_name'] != base_simulation_name:
                comparison_results = {}
                for key in base_simulation:
                    if key != 'simulation_name':
                        if(base_simulation[key]==0):
                            comparison_results[key] = 0
                        else:
                            percentage_change = ((scenario[key] - base_simulation[key]) / base_simulation[key]) * 100
                            comparison_results[key] = percentage_change
                
                # Store the results
                results[f"{base_simulation_name}_vs_{scenario['simulation_name']}"] = comparison_results
        base_path = os.path.dirname(__file__)
        file_path = os.path.join(base_path, "..", "output", "comparison.json")        
        # Save the results to a JSON file
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        #print("The comparison results have been saved to comparison_results.json")

#------------------------------------------------------------------------------------------------------------
def compare_simulations(simulation_file):
    """
    Load simulation data from a JSON file, compare the last two simulations,
    and save percentage changes into comparison.json.
    
    Parameters
    ----------
    simulation_file : str
        Path to the simulation.json file
    """

    # Load simulation data
    with open(simulation_file, "r") as f:
        simulation_data = json.load(f)

    # Ensure at least 2 simulations exist
    if len(simulation_data) < 2:
        raise ValueError("At least two simulations are required for comparison.")

    # Take the last two scenarios
    base_simulation = simulation_data[-2]
    latest_simulation = simulation_data[-1]

    comparison_results = {}
    for key in base_simulation:
        if key != "simulation_name":
            base_val = base_simulation[key]
            latest_val = latest_simulation[key]

            if isinstance(base_val, (int, float)) and isinstance(latest_val, (int, float)):
                if base_val == 0:
                    comparison_results[key] = 0
                else:
                    comparison_results[key] = ((latest_val - base_val) / base_val) * 100

    # Build results dictionary
    results = {
        f"{base_simulation['simulation_name']}_vs_{latest_simulation['simulation_name']}": comparison_results
    }

    # Save to comparison.json (always overwrite)
    base_path = os.path.dirname(simulation_file)
    file_path = os.path.join(base_path, "comparison.json")
    with open(file_path, "w") as f:
        json.dump(results, f, indent=4)

    return results
#-----------------------------------------------------------------------------------------------------
def json_export(schema, data_in, data_out):
    schema_file_path = "metadata/" + schema +".json"
    data_file_path = "metadata/"+ data_in +".json"
    base_path = os.path.dirname(__file__)
    output_file_path = data_out+".json"
    file_path = os.path.join(base_path, "..", "output", output_file_path)        
    
    
    # Load metadata (schema)
    with open(schema_file_path, "r") as schema_file:
        metadata = json.load(schema_file)
    
    # Load simulation data
    with open(data_file_path, "r") as data_file:
        simulation_data = json.load(data_file)
    
    # Combine metadata and data dynamically
    merged_json = {
        "metadata": metadata,  # Add metadata dynamically
        "data": simulation_data
    }
    
    # Save to a new JSON file
    with open(file_path, "w") as outfile:
        json.dump(merged_json, outfile, indent=4)
    
    #print(f"Merged JSON saved to {file_path}")
#-----------------------------------------------------------------
def create_inventory_dataframe(inventory_data):
    # Convert the start date string to a datetime object
        
    # Create a list of dates corresponding to the inventory data
    weeks = list(range(len(inventory_data.inventory_OH)))
    
    # Create a dataframe with the dates and inventory levels
    df = pd.DataFrame({
        "week": weeks,
        "inventory_OH": inventory_data.inventory_OH,
        "inventory_IT": inventory_data.inventory_IT,
        "inventory_FI_E3": inventory_data.inventory_FI[0],
        "inventory_FI_C3": inventory_data.inventory_FI[1]
    })
    return df

#----------------------------------------------------
def write_interface_results(simulation_results_PI,base_simulation_name):
   
    name='output/simulation'    
    
    simulation_data=write_output_json(simulation_results_PI,name + '.json')
    write_output_csv(simulation_results_PI,name + '.csv')
   

    #base_simulation_name="AS_IS"
    compare_simulations(simulation_data, base_simulation_name)
    schema="comparison_schema"
    data_in="comparison"
    data_out="comparison_results"
    json_export(schema, data_in, data_out)

   
  


 

 