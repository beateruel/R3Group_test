# %%
#minutes

import random
from random import randint
import statistics
import time
from datetime import date
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import simpy 

from lib import MTS_Orders as mt


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


from lib import GLN_IO as gl


class SupplyChain(object):
    def __init__(self, env, num_shopfloor,num_logistic):   
            
        self.shopfloor = simpy.Resource(env, num_shopfloor)
        self.logistic = simpy.Resource(env, num_logistic)
        self.inventory_OH_level = simpy.Container(env,capacity=10000,init=0) #kg of the material

        self.weekly_order_cycle_time = np.zeros([52])
        self.weekly_order_production_time = np.zeros([52]) 
        self.weekly_setup_time = np.zeros([52])  
        self.total_costs = 0
        self.inventory_costs = 0
        self.production_costs = 0
        self.distribution_costs = 0
        self.total_emissions=0
        self.inventory_emissions = 0
        self.production_emissions = 0
        self.distribution_emissions = 0
        self.total_income = 0
        self.total_profit = 0      
        
        
    
    def inventory_request(self, env, product_BOM,df_weekly_demand,week,product_name):
        #every week, the production consumes the required material

        week2=int(env.now/(24*7))
        MTS_order.total_orders+=1
        df_demand = df_weekly_demand[(df_weekly_demand['Week']==week) & (df_weekly_demand['ProductName']==product_name) ]
        demand=df_demand['Quantity'].iloc[0]
      
        quantity_PBT = product_BOM[product_BOM["MaterialId"]=="1"]
        amount_PBT_aux = quantity_PBT['MaterialTotalWeight']
        amount_PBT = demand*amount_PBT_aux.iloc[0]/1000
        if(self.inventory_OH_level.level>amount_PBT):
            yield self.inventory_OH_level.get(amount_PBT)
        else:
            MTS_order.failed_orders+=1
            MTS_order.stockouts[week]=1
     
        MTS_order.inventory_OH[week]=int(self.inventory_OH_level.level)        
        yield env.timeout(5)  
    
    def inventory_monitoring_Periodic_min_level(self, env,df_monthly_demand,PBT_price, week,init_week, end_week):  
        #periodic review (T,S) up to level: https://towardsdatascience.com/inventory-management-for-retail-periodic-review-policy-4399330ce8b0
        #order every T periods quantity S
        #S = max quantity - OHI, with S>= min order level 
    
                    
        if(week>0):
            MTS_order.inventory_IT[week]=MTS_order.inventory_IT[week-1]
        else:
            MTS_order.inventory_IT[0]=MTS_order.weekly_purchase_quantity[0]        
        if (week % MTS_order.reorder_point==0):
          #  print(f"comprar material semana {week}")
            
            MTS_order.purchase(df_products_material,df_monthly_demand, PBT_price, int(week/4)+3,ordering_cost, holding_rate)
            MTS_order.material_shortage(week, init_week, end_week)
            #add IT inventory
            MTS_order.inventory_IT[week]+= MTS_order.weekly_purchase_quantity[week] 
          
            
        if (week==MTS_order.delivery_time_que[0]):
     
            MTS_order.delivery_time_que.pop(0)
            quantity= MTS_order.delivery_quantity_que.pop(0)
           
            if quantity>1.0:
                if(MTS_order.inventory_IT[week]-quantity>=0):
                    MTS_order.inventory_IT[week]-=quantity   
                    yield self.inventory_OH_level.put(quantity)
                    MTS_order.inventory_OH[week]=int(self.inventory_OH_level.level)                    
                else:
                    yield self.inventory_OH_level.put(MTS_order.inventory_IT[week])                  
                    MTS_order.inventory_IT[week]=0                   
                    MTS_order.inventory_OH[week]=int(self.inventory_OH_level.level)                    
            else:               
                MTS_order.inventory_IT[week]-=0
                MTS_order.inventory_OH[week]=int(self.inventory_OH_level.level)
             
                                  
            #print(MTS_order.delivery_time_que)
            #print(MTS_order.delivery_quantity_que)
                      
        yield env.timeout(15)     

    def production(self, env, product_BOM, df_weekly_demand, week, product_name):  
        index=1
        production_time_process=0
        df_demand = df_weekly_demand[(df_weekly_demand['Week']==week) & (df_weekly_demand['ProductName']==product_name) ]
        demand=df_demand['Quantity'].iloc[0]
          
        production_rate = product_BOM['ProductionRate']
        cycle_time = product_BOM['CycleTime']
        safety_stock=product_BOM['SafetyStock']
        batch_size = product_BOM['BatchSize'] 
        
        if(product_name=="E3"): 
            index=0
                    
        number_of_batches = math.ceil ((demand + safety_stock - MTS_order.inventory_FI[index,week-1])/batch_size) 
        MTS_order.inventory_FI[index,week]= MTS_order.inventory_FI[index,week-1]
        if(number_of_batches >0):
           # print(f'prouct name {product_name} demand {demand}, number of batches {number_of_batches}')
            working_orders = int (number_of_batches*batch_size)
            
            num_cycles = working_orders/production_rate
            production_time_process = (cycle_time*num_cycles)/3600 #cycle time in seconds, transform to hours
            production_time_process+= (product_BOM['SetupTime']*number_of_batches+product_BOM['QualityCheckTime'])/60 #setup time and quality check in minutes
            self.weekly_setup_time[week]+=number_of_batches*product_BOM['SetupTime']
                   
            if(MTS_order.stockouts[week]==0):           
                self.production_costs+=production_time_process.iloc[0]*labour_cost
                MTS_order.inventory_FI[index,week]+=working_orders 
              
              #  print( MTS_order.inventory_FI)
                MTS_order.production_time[week]+=production_time_process.iloc[0]
                MTS_order.total_holding_cost +=product_price*demand/2
                MTS_order.total_holding_cost = MTS_order.total_holding_cost/2
                yield env.timeout(int(production_time_process.iloc[0])+1)
            else:
                MTS_order.production_downtime[week]+=production_time_process.iloc[0]
              
        
            #print(f'week {week} Production: prouct name {product_name} demand {demand},finished inventory {MTS_order.inventory_FI[index,week]}')
            
        return env.now


   

    def distribution(self, env,df_weekly_demand, week, product_name1, product_name2):

                     
        distribution_time=client_transport_time+client_storage_time        
        df_demand1 = df_weekly_demand[(df_weekly_demand['Week']==week) & (df_weekly_demand['ProductName']==product_name1) ]
        demand1=df_demand1['Quantity'].iloc[0]

        df_demand2 = df_weekly_demand[(df_weekly_demand['Week']==week) & (df_weekly_demand['ProductName']==product_name2) ]
        demand2=df_demand2['Quantity'].iloc[0]
        
        if(MTS_order.stockouts[week]==0):
            MTS_order.distribution_time[week]+=distribution_time
            self.distribution_costs += client_distribution_cost
            self.distribution_emissions += client_distribution_emissions 
            
            if(((MTS_order.inventory_FI[0,week]+MTS_order.inventory_FI[1,week])-(demand1+demand2))>0):              
                self.total_income +=(demand1*product_price+demand2*product_price)
                MTS_order.inventory_FI[0,week]-=demand1
                MTS_order.inventory_FI[1,week]-=demand2 
                #print(f"remaining stock {(MTS_order.inventory_FI[0,week]+MTS_order.inventory_FI[1,week])}")
            else:
                MTS_order.not_in_full[week]=MTS_order.inventory_FI[0,week]+MTS_order.inventory_FI[1,week]-(demand1+demand2)
                self.total_income +=(MTS_order.inventory_FI[0,week]*product_price + MTS_order.inventory_FI[1,week]*product_price)
                MTS_order.inventory_FI[0,week]=0
                MTS_order.inventory_FI[1,week]=0
                #print(f"no stock{((MTS_order.inventory_FI[0,week]+MTS_order.inventory_FI[1,week])-(demand1+demand2))}")
            #print(f'week {week} Distribution: prouct name {product_name1} demand {demand1} ,inventory after distribution { MTS_order.inventory_FI[0,week]}y {product_name2}  demand {demand2} ,inventory after distribution { MTS_order.inventory_FI[1,week]}')
            yield env.timeout(int(distribution_time)+1)
        return env.now


    def KPIs_calculation(self):
        self.inventory_costs=MTS_order.distribution_costs + MTS_order.total_purchase_cost + MTS_order.total_ordering_cost +MTS_order.total_holding_cost
        self.total_costs = self.inventory_costs + supply_chain.production_costs + self.distribution_costs
        self.total_emissions =   self.distribution_emissions  
        self.total_profit = self.total_income-self.total_costs   
      
       
    
def simulation_PI(supply_chain, simulation_name, variation,init_shortage,end_shortage):

    column_names = ['simulation_name', 'demand_variation', 'init_shortage', 'end_shortage','total_costs_euros', 'total_income_euros', 'total_profit_euros', 'total_logistic_costs_euros', 
                    'total_production_costs_euros', 'total_distribution_costs_euros', 'total_purchasing_costs_euros', 'total_ordering_costs_euros', 
                    'total_holding_costs_euros', 'total_emissions(kgCO2e)', 'total_distribution_emissions(kgCO2e)',
                    'production_capacity_utilization(%)', 'downtime_rate(%)','setup_time(%)','OTIF(%)', 'stockouts(%)', 'out_of_time(%)', 'not_in_full(%)']
                    
   
    stockouts=round(100*MTS_order.stockouts.mean(),0)

    overstock_rate= round(MTS_order.KPI_overstock_rate(),0)
    supply_chain.inventory_costs=supply_chain.distribution_costs + MTS_order.total_purchase_cost + MTS_order.total_ordering_cost +MTS_order.total_holding_cost
    supply_chain.total_costs = supply_chain.inventory_costs + supply_chain.production_costs + supply_chain.distribution_costs
    supply_chain.total_profit = supply_chain.total_income - supply_chain.total_costs
    supply_chain.total_emissions =   supply_chain.distribution_emissions      
    
    

    average_production_capacity_utilization =round(np.mean(100*MTS_order.production_time/ (24*7-(client_transport_time+ client_storage_time))),0)
    average_production_downtime = round(np.mean(100*MTS_order.production_downtime/ (24*7-(client_transport_time+ client_storage_time))),0)
    average_setup_time = round(np.mean(100*supply_chain.weekly_setup_time/ (24*7*60)),0)
    downtime_rate =  round(MTS_order.KPI_unplanned_downtime_rate()*100,0)
    aux = (24*7) - (supply_chain.weekly_order_cycle_time)
    out_of_time=round(100*(1-(aux>0).mean()),0)
    not_in_full = round(np.count_nonzero(MTS_order.not_in_full),0)
  
    OTIF =round(100*(np.logical_and((aux>0), (MTS_order.stockouts==0))).mean(),0)
    
   
    valores = np.array([[
    simulation_name, int(variation), int(init_shortage), int(end_shortage),
    int(supply_chain.total_costs), int(supply_chain.total_income),
    int(supply_chain.total_profit), int(supply_chain.inventory_costs),
    int(supply_chain.production_costs), int(supply_chain.distribution_costs),
    int(MTS_order.total_purchase_cost), int(MTS_order.total_ordering_cost),
    int(MTS_order.total_holding_cost), int(supply_chain.total_emissions),
    int(supply_chain.distribution_emissions),
    int(average_production_capacity_utilization), 
    int(average_production_downtime), 
    int(average_setup_time), 
    int(OTIF), 
    int(stockouts), 
    int(out_of_time), 
    int(not_in_full)]], dtype=object)  # Use dtype=object to allow mixed types
             
    simulation_results_PI = pd.DataFrame(valores,columns=column_names )  
    
    
    return simulation_results_PI
     
def run_production_order(env, supply_chain,df_weekly_demand,week, product_name):     
  
            
    #df_demand = df_weekly_demand[(df_weekly_demand['Week']==week) & (df_weekly_demand['ProductName']==product_name) ]
    product_BOM = df_products_material[(df_products_material["ProductName"]==product_name)]
    #amount=df_demand['Quantity']
    
    #reserve the raw material
    init=env.now
    with supply_chain.shopfloor.request() as request:#the order generates a request for a shopfloor worker to reserve  material 
        yield request        
        yield env.process(supply_chain.inventory_request(env,product_BOM,df_weekly_demand, week, product_name))

    #reserve the robot for the production of product_name
    with supply_chain.shopfloor.request() as request:#the order generates a request for a shopfloor worker to produce the product 1
        yield request        
        end = yield env.process(supply_chain.production(env,product_BOM,df_weekly_demand, week, product_name))   
    supply_chain.weekly_order_production_time[week]+=end-init
    
    return env.now


def run_distribution_order(env, supply_chain,df_weekly_demand,week,product_name1,product_name2): 

    #once the production is done the FI is sent to the customer
    with supply_chain.logistic.request() as request:#the order generates a request for a shopfloor worker to produce the product 1
        yield request        
        yield env.process(supply_chain.distribution(env,df_weekly_demand, week,product_name1,product_name2))
    
    return env.now
         

def run_inventory_control(env, supply_chain,df_monthly_demand,df_weekly_demand, PBT_price, week,inventory_policy,init_week, end_week):
    
    if (inventory_policy=="1"):
    
        #sourcing material: periodic review fixed quantity
        #print("other policy")
        a=1          
            
    else:
           
         #sourcing material: periodic review up to level
            with supply_chain.logistic.request() as request:#the order generates a request for a logistic worker to reserve  material 
                yield request                         
                yield env.process(supply_chain.inventory_monitoring_Periodic_min_level(env,df_monthly_demand, PBT_price, week,init_week, end_week))
   
def run_supplychain(env,supply_chain,df_monthly_demand,df_weekly_demand,PBT_price,init_week, end_week,policy):
    
    
    WeeksYear =53
    week=0   
    
    
       
    while week < WeeksYear:
        init=env.now
      #  print(f"inicio bucle {init/(24*7)}")
        
        env.process(run_inventory_control(env, supply_chain,df_monthly_demand,df_weekly_demand, PBT_price, week,policy,init_week, end_week)) 
        #yield env.timeout(1)
        init_p1 = env.now
        product_name1='E3'
        end_p1=yield env.process(run_production_order(env, supply_chain,df_weekly_demand, week,product_name1)) 
                      
        product_name2='C3'
        yield env.process(run_production_order(env, supply_chain,df_weekly_demand, week,product_name2))
       

        end_p2 = yield env.process(run_distribution_order(env, supply_chain,df_weekly_demand,week,product_name1,product_name2))
        supply_chain.weekly_order_cycle_time[week]=end_p2-init_p1
     
        end=env.now
        idle_time =24*7- (end-init)
        if(idle_time<0): 
            idle_time=0
     #   print(f"tiempo ocio {int(idle_time/24)}")
        yield env.timeout(idle_time)  # Wait the difference of hours before the next week orders
       # print('Semana {}, OHI lista gráfico {}'.format(week, inventory.OHI.loc[week,'PBTkg']))
        week+=1
    

    
   
      
  
  
 

# %%
#Steps for main
def main(simulation_name, variation, init_shortage, end_shortage):
   
    policy=2
    
    
    df_monthly_demand = gl.read_monhtly_demand(variation) 
    df_weekly_demand=gl.read_weekly_demand(variation)
  
    PBT_price=gl.read_predicted_supplier_material_prices()    
    MTS_order.initialization(df_supplier_material,df_products_material)
   
    MTS_order.purchase(df_products_material,df_monthly_demand, PBT_price, 0,ordering_cost,holding_rate)
      
    randint(1,55) # Setup. see above
    num_shopfloor,num_logistic=[2,2]   
       
    # Run the simulation
    env = simpy.Environment()
    supply_chain = SupplyChain(env, num_shopfloor,num_logistic)
    env.process(run_supplychain(env,supply_chain,df_monthly_demand,df_weekly_demand,PBT_price, init_shortage, end_shortage,policy))
    env.run(until=12*4*7*24) #counting in hours

  
    #print(
     #   "Running simulation...",
      #  f"\nThe average wait time is {mins} minutes and {secs} seconds.",
        
    #)
   
 
    #print(MTS_order.inventory_IT)
    #print(MTS_order.delivery_time_que)
    #print(MTS_order.delivery_quantity_que)
    #print(MTS_order.inventory_OH)
    #print(MTS_order.inventory_FI)
    MTS_order.inventory_combined_plot(simulation_name)
    df_inventory = gl.create_inventory_dataframe(MTS_order)

    demand1=df_weekly_demand[df_weekly_demand["ProductId"]=="1"]["Quantity"]
    demand2=df_weekly_demand[df_weekly_demand["ProductId"]=="2"]["Quantity"]
    demand=demand1.reset_index(drop=True)+demand2.reset_index(drop=True)
    MTS_order.inventory_individual_plot(demand.head(52), simulation_name)
    df_inventory.to_json('metadata/inventory.json', orient="records", indent=4)
    schema="inventory_schema"
    data_in="inventory"
    data_out="simulation_inventory"
    gl.json_export(schema, data_in, data_out)

    #inventory_valuation = round(MTS_order.KPI_inventory_valuation(demand),0) 
    simulation_results_PI=simulation_PI(supply_chain,simulation_name,variation, init_shortage, end_shortage)
   
    json_result = simulation_results_PI.to_json(orient='records', indent=4)
    print(json_result)

    name=simulation_results_PI.iloc[0]
    name='output/simulation'

     
    #simulation_results_PI.to_json(name + '.json', orient='records', lines= True, mode="a", indent=3)    
    simulation_data=gl.write_output_json(simulation_results_PI,name + '.json')
    gl.write_output_csv(simulation_results_PI,name + '.csv')

    base_simulation_name="AS_IS"
    gl.compare_simulations(simulation_data, base_simulation_name)
    schema="comparison_schema"
    data_in="comparison"
    data_out="comparison_results"
    gl.json_export(schema, data_in, data_out)

    
  
   
if __name__ == "__main__":
    MTS_order = mt.MTS_order()

    df_materials_suppliers = gl.read_material_suppliers()
    #print(df_materials_suppliers)

    df_supplier_material = df_materials_suppliers[(df_materials_suppliers["SupplierId"]=="1") & (df_materials_suppliers["MaterialId"]=="1")]
    df_products_material = gl.read_products_BOM()
    df_products_material = df_products_material[(df_products_material["MaterialId"]=="1")]
    batch_size = df_products_material["BatchSize"]
    reorder_quantity =df_supplier_material["ReorderPoint"]
    print(f"batch size {type(batch_size)}")
    print(f"batch size {type(reorder_quantity)}")
    
    df_main=gl.read_main_parameters()
    df_main_parameters=df_main.T  
    #print(df_main_parameters)
   
    simulation_name = df_main_parameters["SimulationName"].iloc[0]
    demand_variation=df_main_parameters['DemandVariation'].iloc[0]
    init_week_shortage=df_main_parameters['InitWeekShortage'].iloc[0]
    end_week_shortage=df_main_parameters['EndWeekShortage'].iloc[0]
   # price_variation=df_main_parameters['PriceVariation']
    
    #parameters for config
    labour_cost = 17 #average cost/hour in 2023 in Portugal [source = https://ec.europa.eu/eurostat/statistics-explained/index.php?title=Hourly_labour_costs]
    ordering_cost = 100 #€ total cost per order, assumption
    holding_rate = 0.2 #percentage of the products price
    product_price = 0.1 #€ piece
    client_transport_time = 24 #hours, we consider one day for delivery, .
    client_storage_time = 24   #hours, the order is delivered the next day than produced. 
    client_distribution_cost = 100 #transport cost each week 
    client_distribution_emissions = 100 #transport emissions each week  
    
   
       

    simulation_results_PI=main(simulation_name, demand_variation,init_week_shortage,end_week_shortage)
    #gl.write_interface_results(simulation_results_PI)
  

# %%



