from pyomo.environ import * #optimización
import numpy as np
import pandas as pd

def create_dict(df):
    d = {}
    for i in df.index:
        for j in df.columns:
            d[i,j] = df.loc[i,j]
    return d

def T_dict(T, array):

    data = pd.DataFrame(index = T, data={"data":array})["data"].to_dict()

    return data

def create_model(data_model):

    model = ConcreteModel()

    t_s = 1

    ## -- COEFICIENTES VALOR PRESENTE NETO -- ## 
    VPN_F = [round(1/np.power(1+data_model["interest"],i),3) for i in np.arange(1,data_model["lifeyears"]+1)]
    VPN_FS = round(np.sum(VPN_F),3)   

    ## -- VIDA UTIL DEL PROYECTO -- ## 
    model.lifeyears = Param(initialize = data_model["lifeyears"])
    
    ## -- Constante del método linealización big-M -- ## 
    model.big_M = Param(initialize = 1e6)
    
    """
    -----------------------------------------------------------------------------------------
            ----------SETS FIJOS----------
    -----------------------------------------------------------------------------------------
    """ 
    T = range(1,data_model["load"]["len"]+1)     
    model.T = Set(initialize=T)
    model.pv_u = Set(initialize=data_model["pv_modules"]["type"].columns.tolist())
    model.bat_u= Set(initialize=data_model["batteries"]["type"].columns.tolist())
    model.ch_u = Set(initialize=data_model["inverters"]["type"].columns.tolist())
    model.boi_u = Set(initialize=data_model["boilers"]["type"].columns.tolist())
    model.chp_u = Set(initialize=data_model["chps"]["type"].columns.tolist())
    
   
    """
    -----------------------------------------------------------------------------------------
            ----------PARÁMETROS CARACTERÍSTICAS TÉCNICAS DE LOS EQUIPOS----------
    -----------------------------------------------------------------------------------------
    """ 
    model.pv_f = Param(data_model["pv_modules"]["type"].index.to_list(), model.pv_u, initialize = create_dict(data_model["pv_modules"]["type"]), domain = Any)
    model.bat_f = Param(data_model["batteries"]["type"].index.to_list(), model.bat_u, initialize = create_dict(data_model["batteries"]["type"]), domain = Any)
    model.ch_f = Param(data_model["inverters"]["type"].index.to_list(), model.ch_u, initialize = create_dict(data_model["inverters"]["type"]), domain = Any)
    model.boi_f = Param(data_model["boilers"]["type"].index.to_list(), model.boi_u, initialize = create_dict(data_model["boilers"]["type"]), domain = Any)
    model.chp_f = Param(data_model["chps"]["type"].index.to_list(), model.chp_u, initialize = create_dict(data_model["chps"]["type"]), domain = Any)

    """
    -----------------------------------------------------------------------------------------
    ----------GENERACIÓN DE POTENCIA MÓDULOS FOTOVOLTAICOS----------
    -----------------------------------------------------------------------------------------
    """      
    data_model["pv_modules"]["Pmpp"].index = T
    model.p_pv_gen = Param(model.T, model.pv_u, initialize = create_dict(data_model["pv_modules"]["Pmpp"]))

    
    """
    -----------------------------------------------------------------------------------------
    ----------PERFIL DE CARGA ELÉCTRICA----------
    -----------------------------------------------------------------------------------------
    """       
    model.load_el = Param(model.T, initialize= T_dict(T, data_model["load"]["value"]))


    """
    -----------------------------------------------------------------------------------------
            ----------ENERGÍA ELÉCTRICA NO SUMINISTRADA----------
    -----------------------------------------------------------------------------------------
    """    
    model.PEL_NS = Var(model.T, domain=NonNegativeReals) # Variable energía no suministrada
    
    if data_model["ENS_EL"]["active"]:        
        if data_model["ENS_EL"]["type"] == "fixed":            
            model.cost_ens_el = Param(model.T, initialize= T_dict(T, np.repeat(data_model["ENS_EL"]["value"], len(T))))
        elif data_model["ENS_EL"]["type"] == "variable":
            model.cost_ens_el = Param(model.T, initialize= T_dict(T, data_model["ENS_EL"]["value"]))
    else:
        model.cost_ens_el = Param(model.T, initialize= T_dict(T, np.repeat(0, len(T))))
        def None_ENS_EL(m,t):
            return m.PEL_NS[t] == 0
        model.None_ENS_EL = Constraint(model.T, rule=None_ENS_EL)    

    """
    -----------------------------------------------------------------------------------------
            ----------ENERGÍA TÉRMICA NO SUMINISTRADA----------
    -----------------------------------------------------------------------------------------
    """ 

    model.PTH_NS = Var(model.T, domain=NonNegativeReals) # Variable energía no suministrada
    
    if data_model["ENS_TH"]["active"]:        
        if data_model["ENS_TH"]["type"] == "fixed":            
            model.cost_ens_th = Param(model.T, initialize= T_dict(T, np.repeat(data_model["ENS_TH"]["value"], len(T))))
        elif data_model["ENS_TH"]["type"] == "variable":
            model.cost_ens_th = Param(model.T, initialize= T_dict(T, data_model["ENS_TH"]["value"]))
    else:
        model.cost_ens_th = Param(model.T, initialize= T_dict(T, np.repeat(0, len(T))))
        def None_ENS_TH(m,t):
            return m.PTH_NS[t] == 0
        model.None_ENS_TH = Constraint(model.T, rule=None_ENS_TH)      

    
    """
    -----------------------------------------------------------------------------------------
            ----------PARÁMETROS Y VARIABLES RED ELÉCTRICA----------
    -----------------------------------------------------------------------------------------
    """   
    model.PEL_G_L = Var(model.T, domain=NonNegativeReals)                              # VARIABLE Potencia de la red eléctrica a la carga eléctrica
    model.PEL_PV_G = Var(model.ch_u, model.T, domain=NonNegativeReals)                 # VARIABLE Potencia de los PV a la red eléctrica
    model.PEL_G_B = Var(model.ch_u, model.bat_u, model.T, domain=NonNegativeReals)     # VARIABLE Potencia de la red eléctrica a las bateria
    model.PEL_T_G = Var(model.T, domain=NonNegativeReals)                              # VARIABLE Potencia de las turbinas eólicas a la red eléctrica
    
    if data_model["grid"]["active"]:

        model.max_p_grid_el_buy = Param(initialize = data_model["grid"]["pmax_buy"])   # PARÁMETRO Potencia máxima de compra de la red eléctrica
        model.max_p_grid_el_sell = Param(initialize = data_model["grid"]["pmax_sell"]) # PARÁMETRO Potencia máxima de venta a la red eléctrica

        # PARÁMETRO Disponibilidad de la red eléctrica
        if data_model["grid"]["av"]["active"]:
            model.grid_el_av = Param(model.T, initialize = T_dict(T, data_model["grid"]["av"]["value"])) 
        else:
            model.grid_el_av = Param(model.T, initialize = T_dict(T, np.repeat(1, len(T)))) 

        # RESTRICCIÓN Balance y potencia límite de compra red eléctrica
        def PG_lim_rule(m,t):
            return m.PEL_G_L[t] + sum(m.PEL_G_B[tch,tb,t] for tb in m.bat_u for tch in m.ch_u) <= m.grid_el_av[t]*m.max_p_grid_el_buy
        model.PG_lim=Constraint(model.T,rule=PG_lim_rule)
        
        # RESTRICCIÓN Balance y potencia límite de venta red eléctrica
        def PpvG_lim_rule(m,t):
            return sum(m.PEL_PV_G[tch,t] for tch in m.ch_u) + m.PEL_T_G[t] <= m.grid_el_av[t]*m.max_p_grid_el_sell
        model.PpvG_lim=Constraint(model.T,rule=PpvG_lim_rule)

        # PARÁMETRO Precio de compra de energía de la red eléctrica
        if data_model["grid"]["buy_price"]["type"] == "fixed":
            model.price_buy_grid_el = Param(model.T, initialize = T_dict(T, np.repeat(data_model["grid"]["buy_price"]["value"], len(T)))) 
        elif data_model["grid"]["buy_price"]["type"] == "variable":
            model.price_buy_grid_el = Param(model.T, initialize = T_dict(T, data_model["grid"]["buy_price"]["value"])) 
        
        # PARÁMETRO Precio de venta de energía de la red eléctrica
        if data_model["grid"]["sell_price"]["type"] == "fixed":
            model.price_sell_grid_el = Param(model.T, initialize= T_dict(T, np.repeat(data_model["grid"]["sell_price"]["value"], len(T)))) 
        elif data_model["grid"]["sell_price"]["type"] == "variable":
            model.price_sell_grid_el  = Param(model.T, initialize = T_dict(T, data_model["grid"]["sell_price"]["value"])) 

    else:        
        # PARÁMETRO NULO Precio de compra de energía de la red eléctrica
        model.price_buy_grid_el = Param(model.T, initialize = T_dict(T, np.repeat(0, len(T))))     
        # PARÁMETRO NULO Precio de venta de energía de la red eléctrica
        model.price_sell_grid_el = Param(model.T, initialize= T_dict(T, np.repeat(0, len(T))))   

        # RESTRICCIÓN NULA Balance variables de la red eléctrica
        def None_grid(m,t):
            return m.PEL_G_L[t] + sum(m.PEL_G_B[tch,tb,t] for tb in m.bat_u for tch in m.ch_u) + sum(m.PEL_PV_G[tch,t] for tch in m.ch_u) + m.PEL_T_G[t] == 0
        model.None_grid=Constraint(model.T,rule=None_grid)

    
    """
    -----------------------------------------------------------------------------------------
            ----------PARÁMETROS Y VARIABLES DE TURBINAS EÓLICAS----------
    -----------------------------------------------------------------------------------------
    """
    # VARIABLE Número de turbinas eólicas
    model.X_WT = Var(model.wt_u, domain=NonNegativeIntegers)    
    # VARIABLE Potencia de las turbinas eólicas a la carga eléctrica                       
    model.PEL_WT_L = Var(model.T, domain=NonNegativeReals)   
    # VARIABLE Potencia de las turbinas eólicas a las baterías                                      
    model.PEL_WT_B = Var(model.ch_u, model.bat_u, model.T, domain=NonNegativeReals)          
    # VARIABLE Potencia de las turbinas eólicas recortada            
    model.PEL_WT_CUR = Var(model.T, domain=NonNegativeReals)                          

    if data_model["windgen"]["active"]:
        
        # SET Tecnologías de turbinas eólicas
        model.wt_u = Set(initialize=data_model["windgen"]["type"].columns.tolist())   
        # PARÁMETRO Características tecnologías de turbinas eólicas
        model.wt_f = Param(data_model["windgen"]["type"].index.to_list(), model.wt_u, initialize = create_dict(data_model["windgen"]["type"]), domain = Any)
        # PARÁMETRO Generación de turbinas eólicas
        model.p_wt_gen = Param(model.T, model.wt_u, initialize = create_dict(data_model["windgen"]["generation"]))        

        # RESTRICCIÓN Balance de potencia turbinas eólicas
        def WT_balance_rule(m,t):
            return (
                m.PEL_WT_L[t] + m.PEL_T_G[t] + sum(m.PEL_WT_B[tch,tb,t] for tch in model.ch_u for tb in m.bat_u) + 
                model.PEL_WT_CUR[t] 
                    == 
                sum(model.X_WT[tt]*m.p_wt_gen[t,tt] for tt in model.wt_u)
            )
        model.WT_balance_rule=Constraint(model.T,rule=WT_balance_rule)

    else:
        # SET NULO Tecnologías de turbinas eólicas
        model.wt_u = Set(initialize=["None"])

        # PARÁMETRO NULO Características tecnologías de turbinas eólicas
        WT_none_df = pd.DataFrame(index=["C_inst", "C_OM_y"], data={"None":[0,0]})
        model.wt_f = Param(WT_none_df.index.to_list(), model.wt_u, initialize = create_dict(WT_none_df), domain = Any)

        # RESTRICCIÓN NULA Balance de potencia turbinas eólicas               
        def None_WT(m,t):
            return m.PEL_WT_L[t] + m.PEL_T_G[t] + sum(m.PEL_WT_B[tch,tb,t] for tch in model.ch_u for tb in m.bat_u) + model.PEL_WT_CUR[t] == 0
        model.None_WT = Constraint(model.T, rule=None_WT)
        
        # RESTRICCIÓN NULA Número de turbinas eólicas 
        def None_WT_num(m,tt):
            return m.X_WT[tt] == 0
        model.None_WT_num = Constraint(model.wt_u, rule=None_WT_num)


    """
    -----------------------------------------------------------------------------------------
            ----------PARÁMETROS Y VARIABLES DEL GENERADOR DE RESPALDO----------
    -----------------------------------------------------------------------------------------
    """ 
    model.PEL_D_L = Var(model.T, domain=NonNegativeReals)
    model.PEL_D_B = Var(model.ch_u, model.bat_u, model.T, domain=NonNegativeReals)                           
    model.Y_D = Var(model.T, within=Binary)     

    if data_model["generator"]["active"]:

        model.d_fuel_cost = Param(initialize = data_model["generator"]["fuel_cost"])
        model.d_cost_inst = Param(initialize = data_model["generator"]["gen_cost"])
        model.d_p_max = Param(initialize = data_model["generator"]["pmax"])
        model.d_f_min = Param(initialize = data_model["generator"]["fmin"])
        model.d_f_max = Param(initialize = data_model["generator"]["fmax"])
        model.d_fm = Param(initialize = data_model["generator"]["fm"])
        model.d_cost_om = Param(initialize = data_model["generator"]["gen_OM_cost"])
        model.d_min_load_porc = Param(initialize = (data_model["generator"]["min_p_load"]/100))          

        if data_model["generator"]["av"]["active"]:
            model.d_av = Param(model.T, initialize = T_dict(T, data_model["generator"]["av"]["value"]))
        else:
            model.d_av = Param(model.T, initialize = T_dict(T, np.repeat(1, len(T)))) 
        
        # RESTRICCIÓN Límite superior potencia generador diesel
        def PD_lim_rule1(m,t):#
            return m.PEL_D_L[t] + sum(m.PEL_D_B[tch,tb,t] for tch in m.ch_u for tb in m.bat_u) <= m.Y_D[t]*(m.d_p_max*m.d_av[t])
        model.PD_lim1=Constraint(model.T,rule=PD_lim_rule1)

        # RESTRICCIÓN Límite inferior potencia generador diesel
        def PD_lim_rule2(m,t):#
            return m.PEL_D_L[t] + sum(m.PEL_D_B[tch,tb,t] for tch in m.ch_u for tb in m.bat_u) >= m.Y_D[t]*(m.d_min_load_porc*m.d_p_max*m.d_av[t])
        model.PD_lim2=Constraint(model.T,rule=PD_lim_rule2)

    else:
        
        model.d_fuel_cost = Param(initialize = 0)
        model.d_cost_inst = Param(initialize = 0)
        model.d_p_max = Param(initialize = 0)
        model.d_f_min = Param(initialize = 0)
        model.d_f_max = Param(initialize = 0)
        model.d_fm = Param(initialize = 0)
        model.d_cost_om = Param(initialize = 0)
        model.d_min_load_porc = Param(initialize = 0)   

        # RESTRICCIÓN NULA Varible binaria generador diesel
        def None_GenOn(m,t):
            return m.Y_D[t] == 0
        model.None_GenOn=Constraint(model.T,rule=None_GenOn)

        # RESTRICCIÓN NULA Funcionamiento generador diesel
        def None_Gen_fun(m,t):
            return m.PEL_D_L[t] + sum(m.PEL_D_B[tch,tb,t] for tch in m.ch_u for tb in m.bat_u) == 0
        model.None_Gen_fun = Constraint(model.T,rule=None_Gen_fun)   

    """
    -----------------------------------------------------------------------------------------
            ----------PARÁMETROS Y VARIABLES DEL CALDERAS----------
    -----------------------------------------------------------------------------------------
    """
    # VARIABLE Número de calderas
    model.X_BOI = Var(model.boi_u, domain=NonNegativeIntegers)  
    # VARIABLE Binaria, la caldera está en funcionamiento
    model.Y_BOI = Var(model.T, model.boi_u, within=Binary) 
    # VARIABLE Potencia primaria de gas hacia las calderas [kWpe]
    model.PPE_GAS_BOI = Var(model.T, model.boi_u, domain=NonNegativeReals) 
    # VARIABLE Potencia termica generada por las calderas [kWth]
    model.PTH_BOI = Var(model.T, model.boi_u,domain=NonNegativeReals)
    # VARIABLE Auxiliar potencia nominal calderas (restricciones Eficiencia de carga parcial) [kWth]
    model.PNOM_BOI_AUX = Var(model.T, model.boi_u,domain=NonNegativeReals)
    
    if data_model["boilers"]["active"]:
        
        # RESTRICCIÓN Límite superior de potencia de salida de la cladera
        def max_p_boilers(m,t,tboi):
            return m.PTH_BOI[t,tboi] <= m.boi_f['P_th_nom',tboi]*m.X_BOI[tboi]
        model.max_p_boilers = Constraint(model.T, model.boi_u, rule=max_p_boilers)

        # RESTRICCIÓN Límite inferior de potencia de salida de la cladera
        def min_p_boilers(m,t,tboi):
            return m.PTH_BOI[t,tboi] + m.big_M*(1-m.Y_BOI[t,tboi]) >= m.boi_f['P_min_porc',tboi]*m.boi_f['P_th_nom',tboi]*m.X_BOI[tboi] 
        model.min_p_boilers = Constraint(model.T, model.boi_u, rule=min_p_boilers)

        # RESTRICCIÓN Conversión a energía térmica considerando eficiencia de carga parcial
        def efficiency_boilers(m,t,tboi):
            return m.PPE_GAS_BOI[t,tboi] == m.boi_f['y_n',tboi]*m.PNOM_BOI_AUX[t,tboi] + m.boi_f['lamd_n',tboi]*m.PTH_BOI[t,tboi] 
        model.efficiency_boilers = Constraint(model.T, model.boi_u, rule=efficiency_boilers)

        # RESTRICCIÓN Restricción 1 potencia nominal auxiliar calderas
        def boilers_nom_aux_1(m,t,tboi):
            return m.PNOM_BOI_AUX[t,tboi] <= m.big_M*m.Y_BOI[t,tboi]
        model.boilers_nom_aux_1 = Constraint(model.T, model.boi_u, rule=boilers_nom_aux_1)

        # RESTRICCIÓN Restricción 2 potencia nominal auxiliar calderas
        def boilers_nom_aux_2(m,t,tboi):
            return m.PNOM_BOI_AUX[t,tboi] <= m.boi_f['P_th_nom',tboi]*m.X_BOI[tboi] 
        model.boilers_nom_aux_2 = Constraint(model.T, model.boi_u, rule=boilers_nom_aux_2)

        # RESTRICCIÓN Restricción 3 potencia nominal auxiliar calderas
        def boilers_nom_aux_3(m,t,tboi):
            return m.PNOM_BOI_AUX[t,tboi] >= m.boi_f['P_th_nom',tboi]*m.X_BOI[tboi] - m.big_M*(1-m.Y_BOI[t,tboi])
        model.boilers_nom_aux_3 = Constraint(model.T, model.boi_u, rule=boilers_nom_aux_3)

    else:

        # SET NULO Tecnologías de calderas
        model.boi_u = Set(initialize=["None"])

        # PARÁMETRO NULO Características tecnologías de calderas
        BOI_none_df = pd.DataFrame(index=["C_inst", "C_OM_kWh"], data={"None":[0,0]})
        model.boi_f = Param(BOI_none_df.index.to_list(), model.boi_u, initialize = create_dict(BOI_none_df), domain = Any)

        # RESTRICCIÓN NULA Variables binarias caldera
        def None_bin_boiler(m,tboi):
            return sum(m.Y_BOI[t,tboi] for t in m.T) + model.X_BOI[tboi] == 0
        model.None_bin_boiler = Constraint(model.boi_u, rule=None_bin_boiler)
        
        # RESTRICCIÓN NULA Funcionamiento caldera
        def None_fun_boiler(m,t,tboi):
            return m.PTH_BOI[t,tboi] + m.PPE_GAS_BOI[t,tboi] + m.PNOM_BOI_AUX[t,tboi] == 0
        model.None_fun_boiler = Constraint(model.T, model.boi_u, rule=None_fun_boiler)

    """
    -----------------------------------------------------------------------------------------
            ----------PARÁMETROS Y VARIABLES DE LOS CHPs----------
    -----------------------------------------------------------------------------------------
    """

    # VARIABLE Número de unidades CHP
    model.X_CHP = Var(model.chp_u, domain=NonNegativeIntegers)  
    # VARIABLE Binaria, el CHP está en funcionamiento
    model.Y_CHP = Var(model.T, model.chp_u, within=Binary) 
    # VARIABLE Potencia primaria de gas hacia los CHP [kWpe]
    model.PPE_GAS_CHP = Var(model.T, model.chp_u, domain=NonNegativeReals) 
    # VARIABLE Potencia eléctrica generada por los CHP [kWth]
    model.PEL_CHP = Var(model.T, model.chp_u,domain=NonNegativeReals)
    # VARIABLE Potencia térmica generada por los CHP [kWth]
    model.PTH_CHP = Var(model.T, model.chp_u,domain=NonNegativeReals)
    # VARIABLE Auxiliar potencia nominal CHPs (restricciones Eficiencia de carga parcial) [kWth]
    model.PNOM_CHP_AUX = Var(model.T, model.chp_u,domain=NonNegativeReals)

    if data_model["chps"]["active"]:
        # RESTRICCIÓN Límite superior de potencia de salida del CHP
        def max_p_chps(m,t,tchp):
            return m.PEL_CHP[t,tchp] <= m.chp_f['P_nom',tchp]*m.X_chp[tchp]
        model.max_p_chps = Constraint(model.T, model.chp_u, rule=max_p_chps)

        # RESTRICCIÓN Límite inferior de potencia de salida del CHP
        def min_p_chps(m,t,tchp):
            return m.PEL_CHP[t,tchp] + m.big_M*(1-m.Y_CHP[t,tchp]) >= m.chp_f['P_min_porc',tchp]*m.chp_f['P_nom',tchp]*m.X_CHP[tchp] 
        model.min_p_chps = Constraint(model.T, model.chp_u, rule=min_p_chps)

        # RESTRICCIÓN Cálculo de energía primaria utilizada por los CHPs considerando eficiencia de carga parcial
        def efficiency_chps_gas(m,t,tchp):
            return m.PPE_GAS_CHP[t,tchp] == m.chp_f['y_n_el',tchp]*m.PNOM_CHP_AUX[t,tchp] + m.chp_f['lamd_n_el',tchp]*m.PEL_CHP[t,tchp] 
        model.efficiency_chps_gas = Constraint(model.T, model.chp_u, rule=efficiency_chps_gas)

        # RESTRICCIÓN Cálculo de energía térmica generada por los CHPs considerando eficiencia de carga parcial
        def efficiency_chps_th(m,t,tchp):
            return m.PTH_CHP[t,tchp] == m.chp_f['y_n_th',tchp]*m.PNOM_CHP_AUX[t,tchp] + m.chp_f['lamd_n_th',tchp]*m.PEL_CHP[t,tchp] 
        model.efficiency_chps_th = Constraint(model.T, model.chp_u, rule=efficiency_chps_th)

        # RESTRICCIÓN Restricción 1 potencia nominal auxiliar calderas
        def chps_nom_aux_1(m,t,tchp):
            return m.PNOM_CHP_AUX[t,tchp] <= m.big_M*m.Y_CHP[t,tchp]
        model.chps_nom_aux_1 = Constraint(model.T, model.chp_u, rule=chps_nom_aux_1)

        # RESTRICCIÓN Restricción 2 potencia nominal auxiliar calderas
        def chps_nom_aux_2(m,t,tchp):
            return m.PNOM_CHP_AUX[t,tchp] <= m.chp_f['P_nom',tchp]*m.X_CHP[tchp] 
        model.chps_nom_aux_2 = Constraint(model.T, model.chp_u, rule=chps_nom_aux_2)

        # RESTRICCIÓN Restricción 3 potencia nominal auxiliar calderas
        def chps_nom_aux_3(m,t,tchp):
            return m.PNOM_CHP_AUX[t,tchp] >= m.chp_f['P_nom',tchp]*m.X_CHP[tchp] - m.big_M*(1-m.Y_CHP[t,tchp])
        model.chps_nom_aux_3 = Constraint(model.T, model.chp_u, rule=chps_nom_aux_3)


    else:
        # SET NULO Tecnologías de CHPs
        model.chp_u = Set(initialize=["None"])

        # PARÁMETRO NULO Características tecnologías de CHPs
        CHP_none_df = pd.DataFrame(index=["C_inst", "C_OM_kWh"], data={"None":[0,0]})
        model.chp_f = Param(CHP_none_df.index.to_list(), model.chp_u, initialize = create_dict(CHP_none_df), domain = Any)

        # RESTRICCIÓN NULA Variables binarias CHPs
        def None_bin_chp(m,tchp):
            return sum(m.Y_CHP[t,tchp] for t in m.T) + model.X_CHP[tchp] == 0
        model.None_bin_chp = Constraint(model.chp_u, rule=None_bin_chp)
        
        # RESTRICCIÓN NULA Funcionamiento CHPs
        def None_fun_chp(m,t,tchp):
            return m.PTH_CHP[t,tchp] + m.PEL_CHP[t,tchp] + m.PPE_GAS_CHP[t,tchp] + m.PNOM_CHP_AUX[t,tchp] == 0
        model.None_fun_chp = Constraint(model.T, model.chp_u, rule=None_fun_chp)

    """
    -----------------------------------------------------------------------------------------
            ----------VARIABLES ENTERAS----------
    -----------------------------------------------------------------------------------------
    """
    model.X_PVs  = Var(model.pv_u, model.ch_u, domain=NonNegativeIntegers)      # Número de strings de paneles solares 
    model.X_PV = Var(model.pv_u, model.ch_u, domain=NonNegativeIntegers)        # Número de paneles solares
    model.X_Bs  = Var(model.bat_u, model.ch_u, domain=NonNegativeIntegers)      # Número de strings de Baterías
    model.X_B  = Var(model.bat_u, model.ch_u, domain=NonNegativeIntegers)       # Número de Baterías
    model.X_CH  = Var(model.pv_u,model.bat_u,model.ch_u, domain=NonNegativeIntegers)   # Número de inversores híbridos


    """
    -----------------------------------------------------------------------------------------
            ----------VARIABLES BINARIAS----------
    -----------------------------------------------------------------------------------------
    """
    model.Y_B_carg = Var(model.ch_u, model.T, within=Binary)                                # load_el efectiva de baterías (1 = load_el) (0 = Descarga/stand-by)
    model.Y_B_desc = Var(model.ch_u, model.T, within=Binary)                                # Descarga efectiva de baterías (1 = Descarga) (0 = load_el/stand-by)
    model.Y_CH = Var(model.pv_u, model.bat_u, model.ch_u, within=Binary)                           # Sólo un tipo de tecnología de baterías y paneles por inversor

    """
    -----------------------------------------------------------------------------------------
            ----------VARIABLES CONTINUAS----------
    -----------------------------------------------------------------------------------------
    """    
    model.PEL_PV_L = Var(model.ch_u, model.T, domain=NonNegativeReals)                        # Potencia del panel dirigida a la carga [kW] 
    model.PEL_PV_CUR = Var(model.ch_u, model.T, domain=NonNegativeReals)                      # Potencia recortada PV [kW]   
    model.PEL_PV_B = Var(model.ch_u, model.bat_u, model.T, domain=NonNegativeReals)                  # Potencia del panel dirigida a los almacenadores [kW]
    model.PEL_B_L = Var(model.ch_u, model.bat_u, model.T, domain=NonNegativeReals)                   # Potencia de los almacenadores dirigida a la carga [kW]
    model.SOC_B = Var(model.bat_u, model.T, domain=NonNegativeReals)                       # Estado de carga de las baterías tipo [kWh]
    model.CAP_B = Var(model.bat_u, model.T, domain=NonNegativeReals)                      # Capacidad de las baterías tipo [kWh]


    
    #----------------------------------------------------------------------#
    ## -- ELEGIR UN SOLO INVERSOR -- ##
    if data_model["inverters"]["flex"]:
        def PV_BATT_onetype_CH(model,tch):
            return sum(model.Y_CH[tpv,tb,tch] for tpv in model.pv_u for tb in model.bat_u) == 1
        model.PV_onetype_CH=Constraint(model.ch_u, rule=PV_BATT_onetype_CH)
    ## -- ELEGIR MÁS DE UN INVERSOR -- ##
    else:
        def PV_BATT_onetype_CH(model):
            return sum(model.Y_CH[tpv,tb,tch] for tpv in model.pv_u for tb in model.bat_u for tch in model.ch_u) == 1
        model.PV_onetype_CH=Constraint(rule=PV_BATT_onetype_CH)

    def Bxch_rule(model,tpv,tb,tch):#
        return model.X_CH[tpv,tb,tch] <= 10e3*model.Y_CH[tpv,tb,tch]
    model.Bxch_rule=Constraint(model.pv_u, model.bat_u, model.ch_u,rule=Bxch_rule)
    #----------------------------------------------------------------------#

    #----------------------------------------------------------------------#
    # Número de strings por tipo de panel e inversor
    def PV_string_rule(m,tpv,tch):#,t para todo t en T
        return m.X_PVs[tpv, tch] <= sum(m.X_CH[tpv,tb,tch] for tb in m.bat_u)*int(m.ch_f['Num_mpp',tch]*m.ch_f['Num_in_mpp',tch]*np.floor(m.ch_f['Idc_max_in',tch]/m.pv_f['Isc_STC',tpv]))
    model.PV_string=Constraint(model.pv_u, model.ch_u, rule=PV_string_rule)

    #Numero de paneles por tipo de panel e inversor
    def PV_num_rule1(m,tpv, tch):
        return m.X_PV[tpv,tch] >= m.X_PVs[tpv, tch]*np.ceil(m.ch_f['V_mpp_inf',tch]/m.pv_f['Vmp_STC',tpv])
    model.PV_num_rule1=Constraint(model.pv_u, model.ch_u, rule=PV_num_rule1)

    def PV_num_rule2(m,tpv, tch):
        return m.X_PV[tpv,tch] <= m.X_PVs[tpv, tch]*np.floor(m.ch_f['Vdc_max_in',tch]/m.pv_f['Voc_max',tpv])
    model.PV_num_rule2=Constraint(model.pv_u, model.ch_u, rule=PV_num_rule2)

    def PV_num_rule3(m, tch):
        return sum(m.X_PV[tpv,tch]*m.pv_f['P_stc',tpv]/1000 for tpv in m.pv_u) <= sum(m.X_CH[tpv,tb,tch] for tb in m.bat_u for tpv in m.pv_u)*m.ch_f['P_max_pv',tch]
    model.PV_num_rule3=Constraint(model.ch_u, rule=PV_num_rule3)
    #----------------------------------------------------------------------#

    #----------------------------------------------------------------------#
    # Número de baterías por tipo de batería e inversor
    def Batt_string_rule(m,tb,tch):
        return m.X_Bs[tb,tch] <= 10e3*sum(m.X_CH[tpv,tb,tch] for tpv in model.pv_u)
    model.Batt_string_rule=Constraint(model.bat_u, model.ch_u,rule=Batt_string_rule)

    def Batt_num_rule(m, tb, tch):
        return m.X_B[tb,tch] == m.X_Bs[tb,tch]*np.floor(m.ch_f['V_n_batt',tch]/m.bat_f['V_nom',tb])
    model.Batt_num_rule=Constraint(model.bat_u, model.ch_u,rule=Batt_num_rule)
    #----------------------------------------------------------------------#

    #----------------------------------------------------------------------#
    #Balance de Potencia PV por inversor
    def PV_balance_rule(m,tch,t):
        return m.PEL_PV_L[tch,t]+ m.PEL_PV_G[tch,t] + sum(m.PEL_PV_B[tch,tb,t] for tb in m.bat_u) + m.PEL_PV_CUR[tch,t] == sum(m.X_PV[tpv,tch]*m.p_pv_gen[t,tpv] for tpv in m.pv_u)
    model.PV_balance=Constraint(model.ch_u, model.T,rule=PV_balance_rule)

    #Balance de Potencia  
    def P_balance_rule(m,t):
        return m.PEL_G_L[t]+ sum(m.PEL_B_L[tch,tb,t] for tb in m.bat_u for tch in m.ch_u) + sum(m.ch_f['n_dcac',tch]*m.PEL_PV_L[tch,t] for tch in m.ch_u) + m.PEL_D_L[t] + m.PEL_WT_L[t] + m.PEL_NS[t] == m.load_el[t]
    model.P_balance=Constraint(model.T,rule=P_balance_rule)
    #----------------------------------------------------------------------#

    #----------------------------------------------------------------------#
    # Cálculo de estado de carga en baterías cada hora
    def Batt_ts_rule(m,tb,t):#
        if t > m.T.first():
            return (m.SOC_B[tb,t] == m.SOC_B[tb,t-1]*(1-m.bat_f['Auto_des',tb]) + sum(m.bat_f['n',tb]*t_s*(m.PEL_PV_B[tch,tb,t] + m.ch_f['n_acdc',tch]*(m.PEL_D_B[tch,tb,t] + m.PEL_G_B[tch,tb,t] + m.PEL_WT_B[tch,tb,t])) -
                    m.PEL_B_L[tch,tb,t]/(m.bat_f['n',tb]*m.ch_f['n_dcac',tch]) for tch in m.ch_u))
        else:
            return m.SOC_B[tb,t] == m.bat_f['Cap_nom',tb]*sum(m.X_B[tb,tch] for tch in m.ch_u)
    model.Batt_ts=Constraint(model.bat_u, model.T,rule=Batt_ts_rule)

    # Capacidad mínima de baterías 
    def Batt_socmin_rule(m,tb,t):#
        return m.SOC_B[tb,t] >= sum(m.X_B[tb,tch] for tch in m.ch_u)*m.bat_f['Cap_inf',tb]
    model.Batt_socmin=Constraint(model.bat_u, model.T,rule=Batt_socmin_rule)

    # Capacidad máxima de baterías 
    def Batt_socmax_rule(m,tb,t):#
        return m.SOC_B[tb,t] <= m.CAP_B[tb,t] 
    model.Batt_socmax=Constraint(model.bat_u, model.T,rule=Batt_socmax_rule)

    # Degradación de la capacidad de las baterías
    def Bcap_rule1(m,tb,t):#
        if t > m.T.first():
            return m.CAP_B[tb,t] == m.CAP_B[tb,t-1] - (m.bat_f['Deg_kwh',tb]/m.bat_f['n',tb])*sum(m.PEL_B_L[tch,tb,t]/m.ch_f['n_dcac',tch] for tch in m.ch_u)
        else:
            return m.CAP_B[tb,t] == m.bat_f['Cap_nom',tb]*sum(m.X_B[tb,tch] for tch in m.ch_u)
    model.Bcap_rule1=Constraint(model.bat_u, model.T,rule=Bcap_rule1)

    # Degradación anual de la capacidad de las baterías
    def Bcap_rule2(m,tb):#
        return m.CAP_B[tb,m.T.first()]-m.CAP_B[tb,m.T.last()] <= 0.2*m.CAP_B[tb,m.T.first()]/m.bat_f['ty',tb]
    model.Bcap_rule2=Constraint(model.bat_u,rule=Bcap_rule2)
    #----------------------------------------------------------------------#

    #----------------------------------------------------------------------#
    # Restricción salida AC del inversor
    def ac_out_ch(m,tch,t):
        return m.PEL_PV_L[tch,t] + sum(m.PEL_B_L[tch,tb,t] for tb in m.bat_u) + m.PEL_PV_G[tch,t] <= m.ch_f['Pac_max_out',tch]*sum(m.X_CH[tpv,tb,tch] for tb in m.bat_u for tpv in m.pv_u)
    model.ac_out_ch = Constraint(model.ch_u, model.T,rule=ac_out_ch)

    # Restricción entrada AC del inversor
    def ac_in_ch(m,tch,tb,t):
        return m.PEL_G_B[tch,tb,t] + m.PEL_WT_B[tch,tb,t] + m.PEL_D_B[tch,tb,t] <= m.ch_f['Pac_max_in',tch]*sum(m.X_CH[tpv,tb,tch] for tpv in m.pv_u)
    model.ac_in_ch = Constraint(model.ch_u, model.bat_u,model.T,rule=ac_in_ch)
    #----------------------------------------------------------------------#

    #----------------------------------------------------------------------#
    def PpvB_lim_rule1(m,tch,tb,t):
        return m.PEL_PV_B[tch,tb,t] + m.PEL_WT_B[tch,tb,t] + m.PEL_G_B[tch,tb,t] + m.PEL_D_B[tch,tb,t] <= m.X_B[tb, tch]*m.bat_f['P_ch',tb]
    model.PpvB_lim_rule1=Constraint(model.ch_u, model.bat_u, model.T,rule=PpvB_lim_rule1)
    
    def PpvB_lim_rule3(m,tch,tb,t):
        return m.PEL_PV_B[tch,tb,t] + m.PEL_WT_B[tch,tb,t] + m.PEL_G_B[tch,tb,t] + m.PEL_D_B[tch,tb,t] <= (m.ch_f['V_n_batt',tch]*m.ch_f['I_max_ch_pv',tch]/1000)*sum(m.X_CH[tpv,tb,tch] for tpv in m.pv_u)
    model.PpvB_lim_rule3=Constraint(model.ch_u,model.bat_u,model.T,rule=PpvB_lim_rule3)

    def PBL_lims_rule1(m,tch,tb,t):
        return m.PEL_B_L[tch,tb,t] <= m.bat_f['P_des',tb]*m.X_B[tb, tch]
    model.PBL_lims_rule1=Constraint(model.ch_u, model.bat_u, model.T,rule=PBL_lims_rule1)

    def PBL_lims_rule2(m,tch,tb,t):
        return m.PEL_B_L[tch,tb,t] <= (m.ch_f['V_n_batt',tch]*m.ch_f['I_max_des',tch]/1000)*sum(m.X_CH[tpv,tb,tch] for tpv in m.pv_u)
    model.PBL_lims_rule2=Constraint(model.ch_u, model.bat_u, model.T,rule=PBL_lims_rule2)
    #----------------------------------------------------------------------#

    #----------------------------------------------------------------------#
    # Batería cargando con PV
    def Ceff_rule1(m,tch,t):#
        return sum(m.PEL_PV_B[tch,tb,t] + m.PEL_WT_B[tch,tb,t] + m.PEL_G_B[tch,tb,t] + m.PEL_D_B[tch,tb,t] for tb in m.bat_u) <= 100e6*m.Y_B_carg[tch,t]
    model.Ceff_rule1=Constraint(model.ch_u,model.T,rule=Ceff_rule1)

    # Batería descargando
    def Deff_rule1(model,tch,t):#
        return sum(model.PEL_B_L[tch,tb,t] for tb in model.bat_u) <= 100e6*model.Y_B_desc[tch,t]
    model.Deff_rule1=Constraint(model.ch_u,model.T,rule=Deff_rule1)

    # Estado único de batería
    def Bstate_rule(model,tch,t):#
        return model.Y_B_carg[tch,t] + model.Y_B_desc[tch,t] <= 1
    model.Bstate_rule=Constraint(model.ch_u,model.T,rule=Bstate_rule)
    #----------------------------------------------------------------------#
   
    
    if data_model["area"]["active"]:
        model.Area = Param(initialize = data_model["area"]["value"])
        def PV_number_rule(m):#
            return sum(m.X_PV[tpv,tch]*m.pv_f['A',tpv]  for tch in m.ch_u for tpv in m.pv_u) + sum(m.X_WT[tt]*m.wt_f['A',tt] for tt in m.wt_u) <= m.Area
        model.PV_number=Constraint(rule=PV_number_rule)


    if data_model["max_invest"]["active"]:
        model.MaxBudget = Param(initialize = data_model["max_invest"]["value"])
        
        def Budget_rule(m):#
            return sum(m.X_PV[tpv,tch]*(m.pv_f['C_inst',tpv]) for tch in m.ch_u for tpv in m.pv_u) + sum(m.X_B[tb,tch]*m.bat_f['C_inst',tb] for tch in m.ch_u for tb in m.bat_u) \
                   + sum(m.X_CH[tpv,tb,tch]*m.ch_f['C_inst',tch] for tpv in m.pv_u for tb in m.bat_u for tch in m.ch_u) + sum(m.X_WT[tt]*m.wt_f['C_inst',tt] for tt in m.wt_u) + model.d_cost_inst <= m.MaxBudget
        model.Budget=Constraint(rule=Budget_rule)

    if data_model["environment"]["active"]:
        model.EnvC = Param(initialize = data_model["environment"]["mu"]*data_model["environment"]["Cbono"]/1e6)
    else:
        model.EnvC = Param(initialize = 0)
    
    def ComputeFirstStageCost(m):
        return (
            sum(m.X_PV[tpv,tch]*(m.pv_f['C_inst',tpv] + VPN_FS*m.pv_f['C_OM_y',tpv]) for tch in m.ch_u for tpv in m.pv_u)
            + sum(m.X_B[tb,tch]*(m.bat_f['C_inst',tb] + VPN_FS*m.bat_f['C_OM_y',tb]) for tch in m.ch_u for tb in m.bat_u)
            + sum(sum(m.X_CH[tpv,tb,tch] for tb in m.bat_u for tpv in m.pv_u)*(m.ch_f['C_inst',tch]  + VPN_FS*m.ch_f['C_OM_y',tch]) for tch in m.ch_u)
            + sum(m.X_WT[tt]*(m.wt_f['C_inst',tt] + VPN_FS*m.wt_f['C_OM_y',tt]) for tt in m.wt_u) 
            + m.d_cost_inst
            + sum(m.X_BOI[tboi]*m.boi_f['C_inst',tboi] for tboi in m.boi_u)
            + sum(m.X_CHP[tchp]*m.chp_f['C_inst',tchp] for tchp in m.chp_u)
            
            + sum(sum(VPN_F[ii-1]*m.ch_f['C_inst',tch]*sum(m.X_CH[tpv,tb,tch] for tb in m.bat_u for tpv in m.pv_u) for ii in np.arange(int(m.ch_f['ty',tch]),m.lifeyears,int(m.ch_f['ty',tch]))) for tch in m.ch_u)
            + sum(sum(VPN_F[ii-1]*m.bat_f['C_inst',tb]*sum(m.X_B[tb,tch] for tch in m.ch_u) for ii in np.arange(int(m.bat_f['ty',tb]),m.lifeyears,int(m.bat_f['ty',tb]))) for tb in m.bat_u)
            + sum(sum(VPN_F[ii-1]*m.wt_f['C_inst',tt]*m.X_WT[tt] for ii in np.arange(int(m.wt_f['ty',tt]),m.lifeyears,int(m.wt_f['ty',tt]))) for tt in m.wt_u)
            + sum(sum(VPN_F[ii-1]*m.boi_f['C_inst',tboi]*m.X_BOI[tboi] for ii in np.arange(int(m.boi_f['ty',tboi]),m.lifeyears,int(m.boi_f['ty',tboi]))) for tboi in m.boi_u)
            + sum(sum(VPN_F[ii-1]*m.chp_f['C_inst',tchp]*m.X_CHP[tchp] for ii in np.arange(int(m.chp_f['ty',tchp]),m.lifeyears,int(m.chp_f['ty',tchp]))) for tchp in m.chp_u)
        )
    model.FirstStage = Expression(rule=ComputeFirstStageCost)

    def ComputeSecondStageCost(m):
        return VPN_FS*sum(
            m.price_buy_grid_el[t]*(m.PEL_G_L[t] + sum(m.PEL_G_B[tch,tb,t] for tb in m.bat_u for tch in m.ch_u))
            + m.d_fuel_cost*(m.d_f_min*m.Y_D[t] + m.d_fm*(m.PEL_D_L[t] + sum(m.PEL_D_B[tch,tb,t] for tch in m.ch_u for tb in m.bat_u))) 
            + m.d_cost_om*m.Y_D[t]
            + sum(m.PTH_BOI[t,tboi]*m.boi_f['C_OM_kWh',tboi] for tboi in m.boi_u)
            + sum(m.PEL_CHP[t,tchp]*m.chp_f['C_OM_kWh',tchp] for tchp in m.chp_u)
            + m.cost_ens_el[t]*m.PEL_NS[t] - m.price_sell_grid_el[t]*(sum(m.ch_f['n_dcac',tch]*m.PEL_PV_G[tch,t] for tch in m.ch_u) + m.PEL_T_G[t])
            - m.EnvC*sum(sum(m.X_PV[tpv,tch]*m.p_pv_gen[t,tpv] for tpv in m.pv_u) - m.PEL_PV_CUR[tch,t] for tch in m.ch_u)
            - m.EnvC*(sum(m.X_WT[tt]*m.p_wt_gen[t,tt] for tt in m.wt_u) - m.PEL_WT_CUR[t]) 
            
        for t in m.T)

    model.SecondStage = Expression(rule=ComputeSecondStageCost)
    #Función objetivo 
    def obj_rule(m):#regla(Función python)
        return  m.FirstStage + m.SecondStage
    model.Obj=Objective(rule=obj_rule,sense=minimize)                  #Objetive=Objetive, maximizar Valor presente neto

    return model
    

