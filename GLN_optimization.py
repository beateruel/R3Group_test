from lib import GLN_IO as gl
import math

def calculate_year_demand(df_monthly_demand, product_name):

    df=df_monthly_demand
    product_data = df[df['ProductName'] == product_name]
    # Select the first 12 months
    product_data_first_12_months = product_data[product_data['Month'] <= 12]
    # Calculate the total (aggregated) Quantity
    total_demand = product_data_first_12_months['Quantity'].sum()  
    
    return total_demand


#----------------------------------------------------------------------------------------------------------------------------
def EBQ(demand_variation):
    
    total_production_capacity = 5760000 #number of pieces per year (see D2.3 Table 13)
    df_monthly_demand = gl.read_monhtly_demand(demand_variation)
    year_demand=calculate_year_demand(df_monthly_demand, 'E3')
    year_demand=2*year_demand #total demand for the two products

    setup_cost=60 #it is 100 but forced to 60 as in the deliverable
    holding_rate=0.5 #it is 0.2 but forced to 0.5 as in the deliverable
    nominator=2*year_demand*setup_cost
    denominator=holding_rate*(1-(year_demand/total_production_capacity))#rate cost of holding one unit and cost of producing a unit
    print(f"\n nominator {nominator}, denominator {denominator}")
    EBQ = round(math.sqrt(nominator/denominator)     )

    
    return EBQ

#----------------------------------------------------------------------------------------------------------------------------
def EOQ(demand_variation):
   
    df_monthly_demand = gl.read_monhtly_demand(demand_variation)
    year_demand=calculate_year_demand(df_monthly_demand, 'E3')
    year_demand=year_demand*2 #total demand for the two products
    material_weight = 3.15 #available in the BOM.json, but to simplify the PoC
    
    nominator=year_demand*material_weight
    print(f"\n Demand {year_demand} Nominator {nominator}")
    denominator=4 #according to the policy they order every three monhts, there are 4 orders per year
    EOQ = nominator/denominator 
    EOQ=round(EOQ/1000)

    
    return EOQ


 