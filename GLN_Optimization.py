import pandas as pd
import json
import pdb
import numpy as np
import os
import math

def calculate_year_demand(df_monthly_demand, product_name):

    df=df_monthly_demand
    product_data = df[df['ProductName'] == product_name]
    # Select the first 12 months
    product_data_first_12_months = product_data[product_data['Month'] <= 12]
    # Calculate the total (aggregated) Quantity
    total_demand = product_data_first_12_months['Quantity'].sum()

    # Display the aggregated value
    print(f"Total demand for E3 in the first 12 months: {total_demand}")
    
    return total_demand


#----------------------------------------------------------------------------------------------------------------------------
def BEQ(df_monthly_demand, product_name, df_products_material, holding_cost):
   
    year_demand=calculate_year_demand(df_monthly_demand, product_name)
    df_product_data=df_products_material[df_products_material["ProductName"]==product_name]
    nominator=year_demand*df_product_data['SetupCost']*2
    denominator=holding_cost*df_product_data['ProductionCost'] #rate cost of holding one unit and cost of producing a unit

    BEQ = math.sqrt(nominator/denominator)     

    
    return BEQ

#----------------------------------------------------------------------------------------------------------------------------
def EOQ(df_monthly_demand, product_name, df_products_material, holding_cost):
   
    year_demand=calculate_year_demand(df_monthly_demand, product_name)
    df_product_data=df_products_material[df_products_material["ProductName"]==product_name]
    nominator=year_demand*df_product_data['SetupCost']*2
    denominator=holding_cost*df_product_data['ProductionCost'] #rate cost of holding one unit and cost of producing a unit

    BEQ = math.sqrt(nominator/denominator)     

    
    return BEQ


 