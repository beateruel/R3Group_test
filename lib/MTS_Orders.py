import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

class MTS_order(object):
    def __init__(self):   
        
        
        self.material_id=0 #id of the material
        self.supplier_id =0 #id of the supplier       
        self.reorder_point = 0 #number of weeks between two purchases (if fixed)
        self.reoder_quantity = 0 #amount in kg, pieces, litres (if fixed) purchased in each order
        self.min_reorder_quantity = 0 #minimum amount of material purchased in each order
        self.max_reorder_quantity = 0 #maximum amount of matteial can be purchased in each order
        self.lead_time =0 #number of weeks since the purchased order is sent until the material is recevied
        self.delivery_time = 0 #number of weeks between two deliveries
        self.delivery_time_que =[] #list with the pending weeks to deliver the material puchased
        self.delivery_quantity_que=[] #list with the pending purchased material to delivery
        self.inventory_OH = np.zeros(52) #array for saving the inventory on-hand each week
        self.inventory_IT = np.zeros(52) #array for saving the inventory in-transit each week
        self.inventory_FI = np.zeros((2, 52)) #array for saving the inventory in working-progress
        self.stockouts = np.zeros(52)  #array for saving the quantity of material stocked out
        self.not_in_full = np.zeros(52)  #array for saving the quantity not delivered one week
        self.weekly_purchase_quantity = np.zeros(52) #purchase quantity
        self.total_purchase_cost = 0 #purchase cost
        self.total_ordering_cost = 0 #ordering cost
        self.total_holding_cost = 0 #holding cost
        self.total_demand_cost= 0 #monthly demand multiplied with the price of PBT for that month
        self.failed_orders = 0      #number of orders not covered due to insufficient inventory
        self.total_orders = 0
        self.production_time = np.zeros(52)  #time invested for producing the final items each week
        self.distribution_time = np.zeros(52)  #time invested for distributing the final items each week
        self.production_downtime = np.zeros([52]) #time the machine should be working and is stopped
        

        #section for the KPIs related to the inventory
       
        self.inventory_valuation= 0 #
        self.fill_rate= 0           #percentage of orders that cannot be fulfilled due to insufficient inventory  (number of orders that were not completed by the total number of orders in a given period)
        self.stock_outs_rate = 0    #percentage of orders that cannot be fulfilled due to insufficient inventory  (number of orders that were not completed by the total number of orders in a given period)
        self.overstock_rate = 0     #
        self.turnover_rate = 0      #
        
       
        
    def initialization(self, supplier_material,products_material):
             
        self.material_id= supplier_material["MaterialId"].iloc[0]
        self.supplier_id =supplier_material["SupplierId"].iloc[0]      
        self.reorder_point = supplier_material["ReorderPoint"].iloc[0]  #number of weeks between two purchases (if fixed)
        self.reoder_quantiy = supplier_material["ReorderQuantity"].iloc[0]  #amount in kg, pieces, litres (if fixed) purchased in each order
        self.min_reorder_quantity = supplier_material["ReorderQuantity"].iloc[0]  #minimum amount of material purchased in each order
        self.max_reorder_quantity = supplier_material["ReorderQuantity"].iloc[0]  #maximum amount of matteial can be purchased in each order
        self.lead_time =supplier_material["SupplierLeadTime"].iloc[0] #number of weeks since the purchased order is sent until the material is recevied
        self.delivery_time = supplier_material["SupplierDeliveryTime"].iloc[0] #number of weeks between two deliveries
       
    
    def purchase(self,products_material,df_monthly_demand, PBT_price, month,ordering_cost,holding_rate):
        int_deliveries_number= int(self.reorder_point/self.delivery_time)        #number of times the purchased order is delivered betweeb two purchases (supplier.lead_time/frequncy of dlieveries)
        purchase_quantity=0
       # print("Initiating purchasing orders")
       # print(month)
        
        if((len(df_monthly_demand.index)/2)>month):
           # print("Condition satisfied for purchasing")
           # print(len(df_monthly_demand.index)/2)
            period_demand=0
            
            for delivery in range(int_deliveries_number): 
                monthly_demand=0
              
                for e in range(len(products_material)):
                    product_id = products_material.iloc[e]['ProductId']                    
                    df_monthly_product_demand =df_monthly_demand[df_monthly_demand['ProductId']==product_id]
                    material=products_material[products_material['ProductId']==product_id]
                    df_product_demand= df_monthly_product_demand.iloc[delivery+month]                    
                    material_aux=material['MaterialTotalWeight'].iloc[0]               
                    monthly_demand += (df_product_demand['Quantity'] *material['MaterialTotalWeight'].iloc[0] ) /1000

                self.delivery_time_que.append((delivery+month)*self.delivery_time)
                self.delivery_quantity_que.append(monthly_demand)
                purchase_quantity+=monthly_demand
                period_demand +=monthly_demand

                #self.total_holding_cost  += holding_rate*PBT_price[month]*monthly_demand/2 
                #every time there is a delivery or material, the Quantity in the warehouse increases
                #self.total_holding_cost=self.total_holding_cost/2

            if (purchase_quantity<self.min_reorder_quantity):   #if the total material required is lower than the the min reorider quantity         
                self.weekly_purchase_quantity[month*4] = self.min_reorder_quantity
                purchase_quantity-=monthly_demand
                self.delivery_quantity_que.pop(-1)
                self.delivery_quantity_que.append(self.min_reorder_quantity-purchase_quantity) #I will delivery the demand + the material for covering the minimum reorder quantity
            else:
                self.weekly_purchase_quantity[month*4] = purchase_quantity

            self.total_purchase_cost += self.weekly_purchase_quantity[month*4] * PBT_price[month]
            self.total_ordering_cost +=ordering_cost
           
            self.total_demand_cost += period_demand * PBT_price[month]
        
        #    print(f"cantidad comprada {self.weekly_purchase_quantity} y demanda real{period_demand}")
        #    print(f"coste de compra de inventario {self.total_purchase_cost} y coste de demandad real{self.total_demand_cost}")

    def material_shortage(self,week,init_week, end_week):
        int_deliveries_number= int(self.reorder_point/self.delivery_time) 
        
        if init_week <= week <= end_week:
            #print("interrupt material arrival")
            self.weekly_purchase_quantity[week] = 0
            for delivery in range(int_deliveries_number):
                 self.delivery_quantity_que.pop(-1)
            for delivery in range(int_deliveries_number):
                 self.delivery_quantity_que.append(0)
            
           # self.delivery_quantity_que.pop(-1)
           # self.delivery_quantity_que.append(0) #not possible to purchase, material arrival is zero:
           
        
            
    def inventory_individual_plot(self,demand,filename):      
        
        fig=plt.figure(figsize=(5, 5))         
        fig, ax = plt.subplots(5, 1, constrained_layout=True)
        
        Y1 = self.inventory_OH
        Y2 = self.inventory_IT 
        Y3 = self.inventory_FI[0,:]
        Y4 = self.inventory_FI[1,:]    
        X1 = np.arange(0,52)
       
        
        ax[0].plot(X1,Y1, color='red')
        ax[0].set_title("OHI- (kg PBT)") 
        ax[1].plot(X1,Y2,color='grey')
        ax[1].set_title("ITI- (kg PBT)") 
        ax[2].plot(X1,Y3,color='purple')
        ax[2].set_title("FI E3- (number of pieces)") 
        ax[3].plot(X1,Y4,color='pink', linestyle='dashed')
        ax[3].set_title("FI C3- (number of pieces)") 
        ax[4].plot(X1,demand,color='green')
        ax[4].set_title("demand- (number of pieces)")

        base_path = os.path.dirname(__file__)
        # Build the relative path to the output file
        
        filename2=filename+'_individual.png'
        output_path = os.path.join(base_path, "..", "output", filename2)
       
        plt.savefig(output_path)
        plt.show()   
        
    def inventory_combined_plot(self,filename):  

              
        fig2=plt.figure(figsize=(15, 5)) 
        ax = fig2.subplots()
        weeks =  np.arange(0,52)
        ax.set_xlabel("Weeks", fontdict = {'fontsize':14, 'fontweight':'bold', 'color':'tab:blue'})
        ax.set_ylabel(" PBT (kg)")
        ax.set_xlim([1,52])
        ax.set_xticks(range(1, 52))    

        series = [self.inventory_OH, self.inventory_IT]
        colores = ['red', 'grey']
        nombres = [ "OHI - PBT (kg)", "ITI PBT (kg)"]
       
        for serie, color, nombre in zip(series, colores, nombres):       
            ax.plot(weeks,serie, label=nombre, color=color)
        ax.legend()
        base_path = os.path.dirname(__file__)
         # Build the relative path to the output file
        
        filename2=filename+'_combined.png'
        output_path = os.path.join(base_path, "..", "output", filename2)
        plt.savefig(output_path)
        plt.show()

    def inventory_combined_plot_old(self,filename):  

              
        fig2=plt.figure(figsize=(15, 5)) 
        ax = fig2.subplots()
        weeks =  np.arange(0,52)
        ax.set_xlabel("Weeks", fontdict = {'fontsize':14, 'fontweight':'bold', 'color':'tab:blue'})
        ax.set_ylabel("IOH - PBT (kg)")
        ax.set_xlim([1,52])
        ax.set_xticks(range(1, 52))    

        series = [self.inventory_OH, self.inventory_IT,self.inventory_FI[0,:],self.inventory_FI[1,:]]
        colores = ['red', 'grey','purple','pink']
        nombres = [ "OHI PBT (kg)", "ITI PBT (kg)", "FI E3 (pieces)", "FI C3 (pieces)"]
        styles = ["solid", "solid", "solid", "dashed"]
        for serie, color, nombre, styles in zip(series, colores, nombres, styles):       
            ax.plot(weeks,serie, label=nombre, color=color, linestyle=styles)
        ax.legend()
        base_path = os.path.dirname(__file__)
         # Build the relative path to the output file
        
        filename2=filename+'_combined.png'
        output_path = os.path.join(base_path, "..", "output", filename2)
        
        plt.savefig(output_path)
        plt.show() 
        

    
    #KPIS Calculation
    def KPI_inventory_stockouts(self):
    #percentage of stockouts considering failed orders are in combination of the two products oders      
        return (self.failed_orders/self.total_orders)*100

    def KPI_inventory_valuation(self, df_weekly_demand):
        #Inventory valuation is the process of determining the value of a company's inventory at the end of an accounting period.
        #In this case we have followed the Weighted average cost methods (Other LIFO; FIFO; Unit by unit)
        #Total material purchased inventory during the period - total material left after the period. We assume that the rest of material 
        #has been transformed and sold
        #https://sloanreview.mit.edu/article/managing-service-inventory-to-improve-performance/
        total_material_purchased=0
        total_demand=0
  
        for week in range(0,52):
            total_material_purchased+=self.weekly_purchase_quantity[week] #kg of pbt
            total_demand+=df_weekly_demand[week]
            

        total_units_purchased = (total_material_purchased*1000/3.19)
           
        average_cost = self.total_purchase_cost/total_units_purchased
        inventory_valuation = average_cost* (total_units_purchased - total_demand)      
    
        return inventory_valuation


    def KPI_overstock_rate(self):
    #Overstock is the value of the stock you have at hand, that exceeds the estimated demand over your desired sales window.
        overstock_rate = self.total_purchase_cost/ self. total_demand_cost
    
        return overstock_rate

    def KPI_turnover_ratio(self):
    #total purchase cost of inventory minus the total purchase cost of sales
        turnover_ratio =  self.total_demand_cost/self.total_purchase_cost
    
        return turnover_ratio


    def KPI_unplanned_downtime_rate(self):
    #Unplanned downtime= (Time the asset is not working/ Total time) X 100

        total_time = 0 #in hours
        not_working_time = 0 #in hours

        for week in range(0,48):
            total_time+=self.production_time[week]
            if(self.stockouts[week]==1):
                not_working_time+=self.production_time[week]
            
        #print(f'production time array {self.production_time}')
        return not_working_time/total_time


    

