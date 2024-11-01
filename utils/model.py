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

    model.delta_t = Param(initialize = data_model["len_t"]/8760)

    ## -- COEFICIENTES VALOR PRESENTE NETO -- ## 
    VPN_F = [round(1/np.power(1+data_model["interest"],i),3) for i in np.arange(1,data_model["lifeyears"]+1)]
    VPN_FS = round(np.sum(VPN_F),3)   

    ## -- VIDA UTIL DEL PROYECTO -- ## 
    model.lifeyears = Param(initialize = data_model["lifeyears"])
    
    ## -- Constante del método linealización big-M -- ## 
    model.big_M = Param(initialize = 1e6)
    
    """
    -----------------------------------------------------------------------------------------
            ----------SET DE TIEMPO----------
    -----------------------------------------------------------------------------------------
    """ 
    T = range(1,data_model["len_t"]+1)     
    model.T = Set(initialize=T)

    """
    -----------------------------------------------------------------------------------------
    ----------NECECIDADES----------
    -----------------------------------------------------------------------------------------
    """
    if data_model["needs"]["active"]:   

        # SET de necesidades
        model.needs = Set(initialize = data_model["needs"]["load"].columns.to_list())
        # SET de tipos de energía
        model.type_energy = Set(initialize = data_model["needs"]["n"].columns.to_list())        

        # PARÁMETRO Serie temporal de carga de necesidades por periodo de tiempo en kW    
        data_model["needs"]["load"].index = T        
        model.needs_load = Param(model.T, model.needs, initialize = create_dict(data_model["needs"]["load"]), domain = Any)
        # PARÁMETRO Eficiencias de alimentar la nececidad nds desde la fuente de energía toe
        model.needs_n = Param(model.needs, model.type_energy, initialize = create_dict(data_model["needs"]["n"]), domain = Any)       
        
        # VARIABLE Binaria que indica de qué fuente se alimenta la necesidad nds
        model.Y_NEEDS = Var(model.needs, model.type_energy, domain=Binary)
        # VARIABLE Indica la cantidad de carga de la nececidad nds que se alimenta de la fuente toe en el tiempo t [kW]
        model.L_NEEDS = Var(model.T, model.needs, model.type_energy, domain=NonNegativeReals)

        # RESTRICCIÓN Indica qué fuente de energía toe va a demandar la necesidad nds en el instante t
        def balance_power_needs(m,t,nds,toe):
            if m.needs_n[nds, toe] == 0:
                return m.L_NEEDS[t,nds,toe] + m.Y_NEEDS[nds,toe] == 0
            else:
                return m.L_NEEDS[t,nds,toe]*m.needs_n[nds,toe] == m.Y_NEEDS[nds,toe]*m.needs_load[t,nds]        
        model.balance_power_needs = Constraint(model.T, model.needs, model.type_energy, rule=balance_power_needs) 

        # RESTRICCIÓN Limita que la necesidad nds puede ser alimentada por un único tipo de energia toe
        def single_state_energy_needs(m,nds):
            return sum(m.Y_NEEDS[nds,toe] for toe in model.type_energy) == 1
        model.single_state_energy_needs = Constraint(model.needs, rule=single_state_energy_needs) 

    else:
        # SET-NULO de necesidades
        model.needs = Set(initialize=["None"])
        # SET-NULO de tipos de energía
        model.type_energy = model.needs = Set(initialize=["Electrico","Calor","Refrigeracion"])

        model.L_NEEDS = Var(model.T, model.needs, model.type_energy, domain=NonNegativeReals)

        # RESTRICCIÓN-NULA de carga de necesidades
        def needs_none(m,t,nds,toe):
            return m.L_NEEDS[t,nds,toe] == 0
        model.needs_none = Constraint(model.T, model.needs, model.type_energy, rule=needs_none) 

    
    """
    -----------------------------------------------------------------------------------------
    ----------CARGA ELÉCTRICA----------
    -----------------------------------------------------------------------------------------
    """

    def create_ens_el_null():
        model.cost_ens_el = Param(model.T, initialize= T_dict(T, np.repeat(0, len(T))))
        def None_ENS_EL(m,t):
            return m.PEL_NS[t] == 0
        model.None_ENS_EL = Constraint(model.T, rule=None_ENS_EL)    

    # VARIABLE Energía eléctrica no suministrada
    model.PEL_NS = Var(model.T, domain=NonNegativeReals)
    
    if data_model["load_el"]["active"]:
        if data_model["load_el"]["type"] == "fixed": 
            model.load_el = Param(model.T, initialize= T_dict(T, np.repeat(data_model["load_el"]["value"], len(T))))
        elif data_model["load_el"]["type"] == "variable":
            model.load_el = Param(model.T, initialize= T_dict(T, data_model["load_el"]["value"]))
            
        if data_model["load_el"]["ENS_EL"]["active"]:        
            if data_model["load_el"]["ENS_EL"]["type"] == "fixed":            
                model.cost_ens_el = Param(model.T, initialize= T_dict(T, np.repeat(data_model["load_el"]["ENS_EL"]["value"], len(T))))
            elif data_model["load_el"]["ENS_EL"]["type"] == "variable":
                model.cost_ens_el = Param(model.T, initialize= T_dict(T, data_model["load_el"]["ENS_EL"]["value"]))
        else:
            create_ens_el_null()
    else:
        model.load_el = Param(model.T, initialize= T_dict(T, np.repeat(0, len(T))))
        create_ens_el_null()


    """
    -----------------------------------------------------------------------------------------
    ----------CARGA TÉRMICA----------
    -----------------------------------------------------------------------------------------
    """

    #def create_ens_th_null():
    #    model.cost_ens_th = Param(model.T, initialize= T_dict(T, np.repeat(0, len(T))))
    #    def None_ENS_TH(m,t):
    #        return m.PTH_NS[t] == 0
    #    model.None_ENS_TH = Constraint(model.T, rule=None_ENS_TH)    

    # VARIABLE Energía térmica no suministrada
    #model.PTH_NS = Var(model.T, domain=NonNegativeReals)
    
    if data_model["load_th"]["active"]:
        if data_model["load_th"]["type"] == "fixed": 
            model.load_th = Param(model.T, initialize= T_dict(T, np.repeat(data_model["load_th"]["value"], len(T))))
        elif data_model["load_th"]["type"] == "variable":
            model.load_th = Param(model.T, initialize= T_dict(T, data_model["load_th"]["value"]))
            
        #if data_model["load_th"]["ENS_TH"]["active"]:        
        #    if data_model["load_th"]["ENS_TH"]["type"] == "fixed":            
        #        model.cost_ens_th = Param(model.T, initialize= T_dict(T, np.repeat(data_model["load_th"]["ENS_TH"]["value"], len(T))))
        #    elif data_model["load_th"]["ENS_TH"]["type"] == "variable":
        #        model.cost_ens_th = Param(model.T, initialize= T_dict(T, data_model["load_th"]["ENS_TH"]["value"]))
        #else:
        #    create_ens_th_null()
    else:
        model.load_th = Param(model.T, initialize= T_dict(T, np.repeat(0, len(T))))
        #create_ens_th_null()


    """
    -----------------------------------------------------------------------------------------
    ----------CARGA DE REFRIGERACIÓN----------
    -----------------------------------------------------------------------------------------
    """

    #def create_ens_cl_null():
    #    model.cost_ens_cl = Param(model.T, initialize= T_dict(T, np.repeat(0, len(T))))
    #    def None_ENS_CL(m,t):
    #        return m.PCL_NS[t] == 0
    #    model.None_ENS_CL = Constraint(model.T, rule=None_ENS_CL)    

    # VARIABLE Energía de refrigeración no suministrada
    #model.PCL_NS = Var(model.T, domain=NonNegativeReals)
    
    if data_model["load_cl"]["active"]:
        if data_model["load_cl"]["type"] == "fixed": 
            model.load_cl = Param(model.T, initialize= T_dict(T, np.repeat(data_model["load_cl"]["value"], len(T))))
        elif data_model["load_cl"]["type"] == "variable":
            model.load_cl = Param(model.T, initialize= T_dict(T, data_model["load_cl"]["value"]))
            
        #if data_model["load_cl"]["ENS_CL"]["active"]:        
        #    if data_model["load_cl"]["ENS_CL"]["type"] == "fixed":            
        #        model.cost_ens_cl = Param(model.T, initialize= T_dict(T, np.repeat(data_model["load_cl"]["ENS_CL"]["value"], len(T))))
        #    elif data_model["load_cl"]["ENS_CL"]["type"] == "variable":
        #        model.cost_ens_cl = Param(model.T, initialize= T_dict(T, data_model["load_cl"]["ENS_CL"]["value"]))
        #else:
        #    create_ens_cl_null()
    else:
        model.load_cl = Param(model.T, initialize= T_dict(T, np.repeat(0, len(T))))
        #create_ens_cl_null()       


    """
    -----------------------------------------------------------------------------------------
    ----------SETS Y PARÁMETROS TÉCNICOS DE LOS PANELES SOLARES ELÉCTRICOS----------
    -----------------------------------------------------------------------------------------
    """

    if data_model["pv_modules"]["active"] and data_model["inverters"]["active"]:
        
        # SET Tecnologías de paneles solares
        model.pv_u = Set(initialize=data_model["pv_modules"]["type"].columns.tolist()) 
        # PARÁMETRO Características tecnologías de paneles solares
        model.pv_f = Param(data_model["pv_modules"]["type"].index.to_list(), model.pv_u, initialize = create_dict(data_model["pv_modules"]["type"]), domain = Any)

    else:
        # SET-NULO Tecnologías de paneles solares
        model.pv_u = Set(initialize=["None"])
        # PARÁMETRO-NULO Características tecnologías de paneles solares
        PV_none_df = pd.DataFrame(index=["C_inst", "C_OM_y"], data={"None":[0,0]})
        model.pv_f = Param(PV_none_df.index.to_list(), model.pv_u, initialize = create_dict(PV_none_df), domain = Any)

    """
    -----------------------------------------------------------------------------------------
    ----------SETS Y PARÁMETROS TÉCNICOS DE LOS PANELES SOLARES TÉRMICOS ----------
    -----------------------------------------------------------------------------------------
    """ 

    
    if data_model["pv_thermal"]["active"]:
        
        # SET Tecnologías de paneles solares térmicos
        model.pt_u = Set(initialize=data_model["pv_thermal"]["type"].columns.tolist()) 
        # PARÁMETRO Características tecnologías de paneles solares térmicos
        model.pt_f = Param(data_model["pv_thermal"]["type"].index.to_list(), model.pt_u, initialize = create_dict(data_model["pv_thermal"]["type"]), domain = Any)

    else:
        # SET-NULO Tecnologías de paneles solares térmicos
        model.pt_u = Set(initialize=["None"])
        # PARÁMETRO-NULO Características tecnologías de paneles solares térmicos
        PV_none_df = pd.DataFrame(index=["C_inst", "C_OM_y"], data={"None":[0,0]})
        model.pt_f = Param(PV_none_df.index.to_list(), model.pt_u, initialize = create_dict(PV_none_df), domain = Any)


    """
    -----------------------------------------------------------------------------------------
    ----------SETS Y PARÁMETROS TÉCNICOS DE LAS BATERÍAS ----------
    -----------------------------------------------------------------------------------------
    """

    if data_model["batteries"]["active"] and data_model["inverters"]["active"]:
        
        # SET Tecnologías de baterías
        model.bat_u= Set(initialize=data_model["batteries"]["type"].columns.tolist())
        # PARÁMETRO Características tecnologías de baterías
        model.bat_f = Param(data_model["batteries"]["type"].index.to_list(), model.bat_u, initialize = create_dict(data_model["batteries"]["type"]), domain = Any)

    else:
        # SET-NULO Tecnologías de baterías
        model.bat_u = Set(initialize=["None"])
        # PARÁMETRO-NULO Características tecnologías de baterías
        BAT_none_df = pd.DataFrame(index=["C_inst", "C_OM_y"], data={"None":[0,0]})
        model.bat_f = Param(BAT_none_df.index.to_list(), model.bat_u, initialize = create_dict(BAT_none_df), domain = Any)


    """
    -----------------------------------------------------------------------------------------
    ----------VARIABLES, SETS Y PARÁMETROS TÉCNICOS DE LOS INVERSORES HÍBRIDOS ----------
    -----------------------------------------------------------------------------------------
    """

    if data_model["inverters"]["active"] and (data_model["batteries"]["active"] or data_model["pv_modules"]["active"]):
        
        # SET Tecnologías de inversores híbridos
        model.ch_u = Set(initialize=data_model["inverters"]["type"].columns.tolist())
        # PARÁMETRO Características tecnologías de inversores híbridos
        model.ch_f = Param(data_model["inverters"]["type"].index.to_list(), model.ch_u, initialize = create_dict(data_model["inverters"]["type"]), domain = Any)

    else:
        # SET-NULO Tecnologías de inversores híbridos
        model.ch_u = Set(initialize=["None"])
        # PARÁMETRO-NULO Características tecnologías de turbinas eólicas
        CH_none_df = pd.DataFrame(index=["C_inst", "C_OM_y"], data={"None":[0,0]})
        model.ch_f = Param(CH_none_df.index.to_list(), model.ch_u, initialize = create_dict(CH_none_df), domain = Any)

    # VARIABLE Número de inversores híbridos
    model.X_CH  = Var(model.pv_u,model.bat_u,model.ch_u, domain=NonNegativeIntegers)
    # VARIABLE Sólo un tipo de tecnología de baterías y paneles por inversor
    model.Y_CH = Var(model.pv_u, model.bat_u, model.ch_u, within=Binary) 

    


    """
    -----------------------------------------------------------------------------------------
    ----------SETS, PARÁMETROS, VARIABLES Y RESTRICCIONES DE TURBINAS EÓLICAS----------
    -----------------------------------------------------------------------------------------
    """
    if data_model["windgen"]["active"]:
        # SET Tecnologías de turbinas eólicas
        model.wt_u = Set(initialize=data_model["windgen"]["type"].columns.tolist())   
        # PARÁMETRO Características tecnologías de turbinas eólicas
        model.wt_f = Param(data_model["windgen"]["type"].index.to_list(), model.wt_u, initialize = create_dict(data_model["windgen"]["type"]), domain = Any)
    else:
        # SET-NULO Tecnologías de turbinas eólicas
        model.wt_u = Set(initialize=["None"])

        # PARÁMETRO-NULO Características tecnologías de turbinas eólicas
        WT_none_df = pd.DataFrame(index=["C_inst", "C_OM_y"], data={"None":[0,0]})
        model.wt_f = Param(WT_none_df.index.to_list(), model.wt_u, initialize = create_dict(WT_none_df), domain = Any)


    # VARIABLE Número de turbinas eólicas
    model.X_WT = Var(model.wt_u, domain=NonNegativeIntegers)    
    # VARIABLE Potencia de las turbinas eólicas a la carga eléctrica                       
    model.PEL_WT_L = Var(model.T, domain=NonNegativeReals) 
    # VARIABLE Potencia de las turbinas eólicas a los enfriadores eléctricos                  
    model.PEL_WT_EC = Var(model.T, domain=NonNegativeReals)     
    # VARIABLE Potencia de las turbinas eólicas a las baterías                                      
    model.PEL_WT_B = Var(model.ch_u, model.bat_u, model.T, domain=NonNegativeReals)          
    # VARIABLE Potencia de las turbinas eólicas recortada            
    model.PEL_WT_CUR = Var(model.T, domain=NonNegativeReals)   
    # VARIABLE Potencia de las turbinas eólicas a la red eléctrica
    model.PEL_WT_G = Var(model.T, domain=NonNegativeReals)      
    # VARIABLE Potencia de las turbinas eólicas a los calentadores eléctricos
    model.PEL_WT_EH = Var(model.T, domain=NonNegativeReals)                       

    if data_model["windgen"]["active"]:        
        
        # PARÁMETRO Generación de turbinas eólicas
        model.p_wt_gen = Param(model.T, model.wt_u, initialize = create_dict(data_model["windgen"]["generation"]))        

        # RESTRICCIÓN Balance de potencia turbinas eólicas
        def WT_balance_rule(m,t):
            return (
                m.PEL_WT_L[t] + m.PEL_WT_EC[t] + m.PEL_WT_G[t] + sum(m.PEL_WT_B[tch,tb,t] for tch in m.ch_u for tb in m.bat_u) + 
                + m.PEL_WT_EH[t] + m.PEL_WT_CUR[t] 
                    == 
                sum(m.X_WT[tt]*m.p_wt_gen[t,tt] for tt in m.wt_u)
            )
        model.WT_balance_rule=Constraint(model.T,rule=WT_balance_rule)

    else:        

        # RESTRICCIÓN-NULA Balance de potencia turbinas eólicas               
        def None_WT(m,t):
            return m.PEL_WT_L[t] + m.PEL_WT_EC[t] + m.PEL_WT_EH[t] + m.PEL_WT_G[t] + sum(m.PEL_WT_B[tch,tb,t] for tch in m.ch_u for tb in m.bat_u) + m.PEL_WT_CUR[t] == 0
        model.None_WT = Constraint(model.T, rule=None_WT)
        
        # RESTRICCIÓN-NULA Número de turbinas eólicas 
        def None_WT_num(m,tt):
            return m.X_WT[tt] == 0
        model.None_WT_num = Constraint(model.wt_u, rule=None_WT_num)


    """
    -----------------------------------------------------------------------------------------
            ----------PARÁMETROS Y VARIABLES DEL GENERADOR DE RESPALDO----------
    -----------------------------------------------------------------------------------------
    """ 
    # VARIABLE Potencia del generador diesel a la carga 
    model.PEL_D_L = Var(model.T, domain=NonNegativeReals)
    # VARIABLE Potencia del generador diesel a los enfriadores eléctricos
    model.PEL_D_EC = Var(model.T, domain=NonNegativeReals)
    # VARIABLE Potencia del generador diesel a los calentadores eléctricos
    model.PEL_D_EH = Var(model.T, domain=NonNegativeReals)
    # VARIABLE Potencia del generador diesel a las baterías
    model.PEL_D_B = Var(model.ch_u, model.bat_u, model.T, domain=NonNegativeReals)                           
    # VARIABLE Binaria, el generador diesel está en funcionamiento
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
            return m.PEL_D_L[t] + m.PEL_D_EC[t] + m.PEL_D_EH[t] + sum(m.PEL_D_B[tch,tb,t] for tch in m.ch_u for tb in m.bat_u) <= m.Y_D[t]*(m.d_p_max*m.d_av[t])
        model.PD_lim_rule1 = Constraint(model.T, rule=PD_lim_rule1)

        # RESTRICCIÓN Límite inferior potencia generador diesel
        def PD_lim_rule2(m,t):#
            return m.PEL_D_L[t] + m.PEL_D_EC[t] + m.PEL_D_EH[t] + sum(m.PEL_D_B[tch,tb,t] for tch in m.ch_u for tb in m.bat_u) >= m.Y_D[t]*(m.d_min_load_porc*m.d_p_max*m.d_av[t])
        model.PD_lim_rule2=Constraint(model.T,rule=PD_lim_rule2)

    else:
        
        model.d_fuel_cost = Param(initialize = 0)
        model.d_cost_inst = Param(initialize = 0)
        model.d_p_max = Param(initialize = 0)
        model.d_f_min = Param(initialize = 0)
        model.d_f_max = Param(initialize = 0)
        model.d_fm = Param(initialize = 0)
        model.d_cost_om = Param(initialize = 0)
        model.d_min_load_porc = Param(initialize = 0)   

        # RESTRICCIÓN-NULA Varible binaria generador diesel
        def None_GenOn(m,t):
            return m.Y_D[t] == 0
        model.None_GenOn=Constraint(model.T,rule=None_GenOn)

        # RESTRICCIÓN-NULA Funcionamiento generador diesel
        def None_Gen_fun(m,t):
            return m.PEL_D_L[t] + m.PEL_D_EC[t] + m.PEL_D_EH[t] + sum(m.PEL_D_B[tch,tb,t] for tch in m.ch_u for tb in m.bat_u) == 0
        model.None_Gen_fun = Constraint(model.T,rule=None_Gen_fun)  

    
    """
    -----------------------------------------------------------------------------------------
            ----------PARÁMETROS Y VARIABLES DE LOS ENFRIADORES DE ABSORCIÓN----------
    -----------------------------------------------------------------------------------------
    """ 

    if data_model["abs_chillers"]["active"]:
        # SET Tecnologías de enfriadores de absorción
        model.ac_u = Set(initialize=data_model["abs_chillers"]["type"].columns.tolist())
        # PARÁMETRO Características tecnologías de enfriadores de absorción
        model.ac_f = Param(data_model["abs_chillers"]["type"].index.to_list(), model.ac_u, initialize = create_dict(data_model["abs_chillers"]["type"]), domain = Any)
    else:
        # SET-NULO Tecnologías de enfriadores de absorción
        model.ac_u = Set(initialize=["None"])

        # PARÁMETRO-NULO Características tecnologías de enfriadores de absorción
        AC_none_df = pd.DataFrame(index=["C_inst", "C_OM_kWh"], data={"None":[0,0]})
        model.ac_f = Param(AC_none_df.index.to_list(), model.ac_u, initialize = create_dict(AC_none_df), domain = Any)


    # VARIABLE Número de enfriadores de absorción
    model.X_AC = Var(model.ac_u, domain=NonNegativeIntegers)  
    # VARIABLE Binaria, el enfriador de absorción está en funcionamiento
    model.Y_AC = Var(model.T, model.ac_u, within=Binary)     
    # VARIABLE Potencia de enfriamiento generada por los enfriadores de absorción [kWcl]
    model.PCL_AC = Var(model.T, model.ac_u,domain=NonNegativeReals)
    # VARIABLE Auxiliar potencia nominal enfriadores de absorción (restricciones Eficiencia de carga parcial) [kWcl]
    model.PNOM_AC_AUX = Var(model.T, model.ac_u,domain=NonNegativeReals)
    # VARIABLE Potencia térmica de los CHP a los enfriadores de absorción [kWth]
    model.PTH_CHP_AC = Var(model.T, model.ac_u, domain=NonNegativeReals)
    # VARIABLE Potencia térmica de los calentadores eléctricos a los enfriadores de absorción [kWth]
    model.PTH_EH_AC = Var(model.T, model.ac_u, domain=NonNegativeReals)
    # VARIABLE Potencia térmica de los calentadores eléctricos a los enfriadores de absorción [kWth]
    model.PTH_BOI_AC = Var(model.T, model.ac_u, domain=NonNegativeReals)

    
    if data_model["abs_chillers"]["active"]:
        
        # RESTRICCIÓN Límite superior de potencia de salida del enfriador de absorción
        def max_p_ac(m,t,tac):
            return m.PCL_AC[t,tac] <= m.ac_f['P_cl_nom',tac]*m.X_AC[tac]
        model.max_p_ac = Constraint(model.T, model.ac_u, rule=max_p_ac)

        # RESTRICCIÓN Límite inferior de potencia de salida del enfriador de absorción
        def min_p_ac(m,t,tac):
            return m.PCL_AC[t,tac] + m.big_M*(1-m.Y_AC[t,tac]) >= m.ac_f['P_min_porc',tac]*m.ac_f['P_cl_nom',tac]*m.X_AC[tac] 
        model.min_p_ac = Constraint(model.T, model.ac_u, rule=min_p_ac)

        # RESTRICCIÓN Conversión a energía de enfriamiento considerando eficiencia de carga parcial
        def efficiency_ac(m,t,tac):
            return m.PTH_CHP_AC[t,tac] + m.PTH_EH_AC[t,tac] + m.PTH_BOI_AC[t,tac] == m.ac_f['y_n',tac]*m.PNOM_AC_AUX[t,tac] + m.ac_f['lamd_n',tac]*m.PCL_AC[t,tac] 
        model.efficiency_ac = Constraint(model.T, model.ac_u, rule=efficiency_ac)

        # RESTRICCIÓN Restricción 1 potencia nominal auxiliar enfriadores de absorción
        def ac_nom_aux_1(m,t,tac):
            return m.PNOM_AC_AUX[t,tac] <= m.big_M*m.Y_AC[t,tac]
        model.ac_nom_aux_1 = Constraint(model.T, model.ac_u, rule=ac_nom_aux_1)

        # RESTRICCIÓN Restricción 2 potencia nominal auxiliar enfriadores de absorción
        def ac_nom_aux_2(m,t,tac):
            return m.PNOM_AC_AUX[t,tac] <= m.ac_f['P_cl_nom',tac]*m.X_AC[tac] 
        model.ac_nom_aux_2 = Constraint(model.T, model.ac_u, rule=ac_nom_aux_2)

        # RESTRICCIÓN Restricción 3 potencia nominal auxiliar enfriadores de absorción
        def ac_nom_aux_3(m,t,tac):
            return m.PNOM_AC_AUX[t,tac] >= m.ac_f['P_cl_nom',tac]*m.X_AC[tac] - m.big_M*(1-m.Y_AC[t,tac])
        model.ac_nom_aux_3 = Constraint(model.T, model.ac_u, rule=ac_nom_aux_3)

    else:        

        # RESTRICCIÓN-NULA Variables binarias enfriadores de absorción
        def None_bin_ac(m,tac):
            return sum(m.Y_AC[t,tac] for t in m.T) + m.X_AC[tac] == 0
        model.None_bin_ac = Constraint(model.ac_u, rule=None_bin_ac)
        
        # RESTRICCIÓN-NULA Funcionamiento enfriadores de absorción
        def None_fun_ac(m,t,tac):
            return m.PCL_AC[t,tac] + m.PTH_CHP_AC[t,tac] + m.PTH_EH_AC[t,tac] + m.PNOM_AC_AUX[t,tac] == 0
        model.None_fun_ac = Constraint(model.T, model.ac_u, rule=None_fun_ac)
    
    """
    -----------------------------------------------------------------------------------------
            ----------PARÁMETROS Y VARIABLES DEL CALDERAS----------
    -----------------------------------------------------------------------------------------
    """
    if data_model["boilers"]["active"]:
        model.boi_u = Set(initialize=data_model["boilers"]["type"].columns.tolist())
        model.boi_f = Param(data_model["boilers"]["type"].index.to_list(), model.boi_u, initialize = create_dict(data_model["boilers"]["type"]), domain = Any)
    else:
        # SET-NULO Tecnologías de calderas
        model.boi_u = Set(initialize=["None"])

        # PARÁMETRO-NULO Características tecnologías de calderas
        BOI_none_df = pd.DataFrame(index=["C_inst", "C_OM_kWh"], data={"None":[0,0]})
        model.boi_f = Param(BOI_none_df.index.to_list(), model.boi_u, initialize = create_dict(BOI_none_df), domain = Any)    


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
    # VARIABLE Potencia termica de las calderas a la carga [kWth]
    model.PTH_BOI_L = Var(model.T, domain=NonNegativeReals)
    
    if data_model["boilers"]["active"]:        
        
        # RESTRICCIÓN Límite superior de potencia de salida de la caldera
        def max_p_boilers(m,t,tboi):
            return m.PTH_BOI[t,tboi] <= m.boi_f['P_th_nom',tboi]*m.X_BOI[tboi]
        model.max_p_boilers = Constraint(model.T, model.boi_u, rule=max_p_boilers)

        # RESTRICCIÓN Límite inferior de potencia de salida de la caldera
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

        # RESTRICCIÓN Balance de potencia térmica de las calderas
        def PTH_balance_boi(m,t):
            return (
                sum(m.PTH_BOI[t,tboi] for tboi in m.boi_u) ==
                sum(m.PTH_BOI_AC[t,tac] for tac in m.ac_u) # A los enfriadores de absorsión
                + m.PTH_BOI_L[t]
            )
        model.PTH_balance_boi = Constraint(model.T, rule=PTH_balance_boi)

    else:        

        # RESTRICCIÓN-NULA Variables binarias caldera
        def None_bin_boiler(m,tboi):
            return sum(m.Y_BOI[t,tboi] for t in m.T) + model.X_BOI[tboi] == 0
        model.None_bin_boiler = Constraint(model.boi_u, rule=None_bin_boiler)
        
        # RESTRICCIÓN-NULA Funcionamiento caldera
        def None_fun_boiler(m,t):
            return (
                sum(m.PTH_BOI[t,tboi] + m.PPE_GAS_BOI[t,tboi] + m.PNOM_BOI_AUX[t,tboi] for tboi in m.boi_u)
                + sum(m.PTH_BOI_AC[t,tac] for tac in m.ac_u) + m.PTH_BOI_L[t]
                == 0
            )
        model.None_fun_boiler = Constraint(model.T, rule=None_fun_boiler)      
        

    """
    -----------------------------------------------------------------------------------------
            ----------PARÁMETROS Y VARIABLES DE LOS CHPs----------
    -----------------------------------------------------------------------------------------
    """

    if data_model["chps"]["active"]:
        # SET Tecnologías de CHPs
        model.chp_u = Set(initialize=data_model["chps"]["type"].columns.tolist())
        # PARÁMETRO Características tecnologías de CHPs
        model.chp_f = Param(data_model["chps"]["type"].index.to_list(), model.chp_u, initialize = create_dict(data_model["chps"]["type"]), domain = Any)
    else:
        # SET-NULO Tecnologías de CHPs
        model.chp_u = Set(initialize=["None"])

        # PARÁMETRO-NULO Características tecnologías de CHPs
        CHP_none_df = pd.DataFrame(index=["C_inst", "C_OM_kWh"], data={"None":[0,0]})
        model.chp_f = Param(CHP_none_df.index.to_list(), model.chp_u, initialize = create_dict(CHP_none_df), domain = Any)
   

    # VARIABLE Número de unidades CHP
    model.X_CHP = Var(model.chp_u, domain=NonNegativeIntegers)  
    # VARIABLE Binaria, el CHP está en funcionamiento
    model.Y_CHP = Var(model.T, model.chp_u, within=Binary) 
    # VARIABLE Potencia primaria de gas hacia los CHP [kWpe]
    model.PPE_GAS_CHP = Var(model.T, model.chp_u, domain=NonNegativeReals) 
    # VARIABLE Auxiliar potencia nominal CHPs (restricciones Eficiencia de carga parcial) [kWth]
    model.PNOM_CHP_AUX = Var(model.T, model.chp_u,domain=NonNegativeReals)
    
    # VARIABLE Potencia eléctrica generada por los CHP [kWel]
    model.PEL_CHP = Var(model.T, model.chp_u,domain=NonNegativeReals)
    # VARIABLE Potencia eléctrica de los CHP a la carga [kWel]
    model.PEL_CHP_L = Var(model.T, domain=NonNegativeReals)    
    # VARIABLE Potencia eléctrica de los CHP a las baterias [kWel]
    model.PEL_CHP_B = Var(model.ch_u, model.bat_u, model.T, domain=NonNegativeReals)  
    # VARIABLE Potencia eléctrica de los CHP a la red [kWel]
    model.PEL_CHP_G = Var(model.ch_u, model.bat_u, model.T, domain=NonNegativeReals)  
        
    # VARIABLE Potencia térmica generada por los CHP [kWth]
    model.PTH_CHP = Var(model.T, model.chp_u,domain=NonNegativeReals)     
    # VARIABLE Potencia térmica de los CHP a la carga térmica [kWth]
    model.PTH_CHP_L = Var(model.T, domain=NonNegativeReals)
    # VARIABLE Potencia térmica de los CHP no utilizada [kWth]
    model.PTH_CHP_CUR = Var(model.T, domain=NonNegativeReals)

    if data_model["chps"]["active"]:

        # RESTRICCIÓN Límite superior de potencia de salida del CHP
        def max_p_chps(m,t,tchp):
            return m.PEL_CHP[t,tchp] <= m.chp_f['P_nom',tchp]*m.X_CHP[tchp]
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

        # RESTRICCIÓN Restricción 1 potencia nominal auxiliar CHPs
        def chps_nom_aux_1(m,t,tchp):
            return m.PNOM_CHP_AUX[t,tchp] <= m.big_M*m.Y_CHP[t,tchp]
        model.chps_nom_aux_1 = Constraint(model.T, model.chp_u, rule=chps_nom_aux_1)

        # RESTRICCIÓN Restricción 2 potencia nominal auxiliar CHPs
        def chps_nom_aux_2(m,t,tchp):
            return m.PNOM_CHP_AUX[t,tchp] <= m.chp_f['P_nom',tchp]*m.X_CHP[tchp] 
        model.chps_nom_aux_2 = Constraint(model.T, model.chp_u, rule=chps_nom_aux_2)

        # RESTRICCIÓN Restricción 3 potencia nominal auxiliar CHPs
        def chps_nom_aux_3(m,t,tchp):
            return m.PNOM_CHP_AUX[t,tchp] >= m.chp_f['P_nom',tchp]*m.X_CHP[tchp] - m.big_M*(1-m.Y_CHP[t,tchp])
        model.chps_nom_aux_3 = Constraint(model.T, model.chp_u, rule=chps_nom_aux_3)

        # RESTRICCIÓN Balance de potencia térmica de los CHPs
        def PTH_balance_th_chp(m,t):
            return sum(m.PTH_CHP_AC[t,tac] for tac in m.ac_u) + m.PTH_CHP_L[t] + m.PTH_CHP_CUR[t] == sum(m.PTH_CHP[t,tchp] for tchp in m.chp_u)
        model.PTH_balance_th_chp = Constraint(model.T, rule=PTH_balance_th_chp)

        # RESTRICCIÓN Balance de potencia eléctrica de los CHPs
        def PTH_balance_el_chp(m,t):
            return m.PEL_CHP_L[t] + m.PEL_CHP_G[t] + sum(m.PEL_CHP_B[tch,tb,t] for tch in m.ch_u for tb in m.bat_u) == sum(m.PEL_CHP[t,tchp] for tchp in m.chp_u)
        model.PTH_balance_el_chp = Constraint(model.T, rule=PTH_balance_el_chp)

    else:        

        # RESTRICCIÓN-NULA Variables binarias CHPs
        def None_bin_chp(m,tchp):
            return sum(m.Y_CHP[t,tchp] for t in m.T) + model.X_CHP[tchp] == 0
        model.None_bin_chp = Constraint(model.chp_u, rule=None_bin_chp)
        
        # RESTRICCIÓN-NULA Funcionamiento CHPs
        def None_fun_chp(m,t,tchp):
            return m.PTH_CHP[t,tchp] + m.PEL_CHP[t,tchp] + m.PPE_GAS_CHP[t,tchp] + m.PNOM_CHP_AUX[t,tchp] == 0
        model.None_fun_chp = Constraint(model.T, model.chp_u, rule=None_fun_chp)

        # RESTRICCIÓN-NULA Funcionamiento CHPs salidas
        def None_fun_chp_t(m,t):
            return m.PEL_CHP_L[t] + m.PEL_CHP_G[t] + sum(m.PEL_CHP_B[tch,tb,t] for tch in m.ch_u for tb in m.bat_u) + m.PTH_CHP_L[t] + m.PTH_CHP_CUR[t] == 0
        model.None_fun_chp_t = Constraint(model.T, rule=None_fun_chp_t)


    """
    -----------------------------------------------------------------------------------------
    ----------PARÁMETROS, VARIABLES Y RESTRICCIONES DE PANELES SOLARES----------
    -----------------------------------------------------------------------------------------
    """
    # VARIABLE Número de strings de paneles solares 
    model.X_PVs  = Var(model.pv_u, model.ch_u, domain=NonNegativeIntegers) 
    # VARIABLE Número de paneles solares 
    model.X_PV = Var(model.pv_u, model.ch_u, domain=NonNegativeIntegers)   
    # VARIABLE Potencia del panel dirigida a la carga [kWel] 
    model.PEL_PV_L = Var(model.ch_u, model.T, domain=NonNegativeReals)  
    # VARIABLE Potencia PV no utilizada [kWel]                         
    model.PEL_PV_CUR = Var(model.ch_u, model.T, domain=NonNegativeReals)     
    # VARIABLE Potencia del panel dirigida a las baterías [kWel]                 
    model.PEL_PV_B = Var(model.ch_u, model.bat_u, model.T, domain=NonNegativeReals)         
    # VARIABLE Potencia del panel dirigida a los enfriadores eléctricos [kWel] 
    model.PEL_PV_EC = Var(model.ch_u, model.T, domain=NonNegativeReals) 
    # VARIABLE Potencia del panel dirigida a los calentadores eléctricos [kWel] 
    model.PEL_PV_EH = Var(model.ch_u, model.T, domain=NonNegativeReals) 
    # VARIABLE Potencia de los PV a la red eléctrica [kWel]                  
    model.PEL_PV_G = Var(model.ch_u, model.T, domain=NonNegativeReals)

    if data_model["pv_modules"]["active"] and data_model["inverters"]["active"]:

        data_model["pv_modules"]["Pmpp"].index = T
        # PARÁMETRO Potencia generada por cada tecnología de panel solar
        model.p_pv_gen = Param(model.T, model.pv_u, initialize = create_dict(data_model["pv_modules"]["Pmpp"]))

        # RESTRICCIÓN Número de strings por tipo de panel e inversor
        def PV_string_rule(m,tpv,tch):
            return m.X_PVs[tpv, tch] <= sum(m.X_CH[tpv,tb,tch] for tb in m.bat_u)*int(m.ch_f['Num_mpp',tch]*m.ch_f['Num_in_mpp',tch]*np.floor(m.ch_f['Idc_max_in',tch]/m.pv_f['Isc_STC',tpv]))
        model.PV_string=Constraint(model.pv_u, model.ch_u, rule=PV_string_rule)

        # RESTRICCIÓN Numero mínimo de paneles por tipo de panel e inversor
        def PV_num_rule1(m,tpv, tch):
            return m.X_PV[tpv,tch] >= m.X_PVs[tpv, tch]*np.ceil(m.ch_f['V_mpp_inf',tch]/m.pv_f['Vmp_STC',tpv])
        model.PV_num_rule1=Constraint(model.pv_u, model.ch_u, rule=PV_num_rule1)

        # RESTRICCIÓN Numero máximo de paneles por tipo de panel e inversor voltaje
        def PV_num_rule2(m,tpv, tch):
            return m.X_PV[tpv,tch] <= m.X_PVs[tpv, tch]*np.floor(m.ch_f['Vdc_max_in',tch]/m.pv_f['Voc_max',tpv])
        model.PV_num_rule2=Constraint(model.pv_u, model.ch_u, rule=PV_num_rule2)

        # RESTRICCIÓN Numero máximo de paneles por tipo de panel e inversor potencia
        def PV_num_rule3(m, tch):
            return sum(m.X_PV[tpv,tch]*m.pv_f['P_stc',tpv]/1000 for tpv in m.pv_u) <= sum(m.X_CH[tpv,tb,tch] for tb in m.bat_u for tpv in m.pv_u)*m.ch_f['P_max_pv',tch]
        model.PV_num_rule3=Constraint(model.ch_u, rule=PV_num_rule3)

        # RESTRICCIÓN Balance de Potencia PV por inversor
        def PV_balance_rule(m,tch,t):
            return m.PEL_PV_EC[tch,t] + m.PEL_PV_EH[tch,t] + m.PEL_PV_L[tch,t]+ m.PEL_PV_G[tch,t] + sum(m.PEL_PV_B[tch,tb,t] for tb in m.bat_u) + m.PEL_PV_CUR[tch,t] == sum(m.X_PV[tpv,tch]*m.p_pv_gen[t,tpv] for tpv in m.pv_u)
        model.PV_balance=Constraint(model.ch_u, model.T,rule=PV_balance_rule)


    else:
        # RESTRICCIÓN-NULA Variables enteras y binarias paneles
        def PV_null_num(m,tpv,tch):
            return m.X_PVs[tpv,tch] + m.X_PV[tpv,tch] == 0
        model.PV_null_num=Constraint(model.pv_u, model.ch_u, rule=PV_null_num)
        # RESTRICCIÓN-NULA Variables reales paneles
        def PV_null_din(m,tch,t):
            return m.PEL_PV_L[tch,t] + m.PEL_PV_CUR[tch,t] + sum(m.PEL_PV_B[tch,tb,t] for tb in m.bat_u) + m.PEL_PV_EC[tch,t] + m.PEL_PV_EH[tch,t] + m.PEL_PV_G[tch,t] == 0
        model.PV_null_din=Constraint(model.ch_u, model.T, rule=PV_null_din)

        


    """
    -----------------------------------------------------------------------------------------
    ----------PARÁMETROS, VARIABLES Y RESTRICCIONES DE LAS BATERÍAS----------
    -----------------------------------------------------------------------------------------
    """
    # VARIABLE Número de strings de Baterías
    model.X_Bs  = Var(model.bat_u, model.ch_u, domain=NonNegativeIntegers) 
    # VARIABLE Número de Baterías
    model.X_B  = Var(model.bat_u, model.ch_u, domain=NonNegativeIntegers) 
    # VARIABLE Carga efectiva de baterías (1 = load_el) (0 = Descarga/stand-by)
    model.Y_B_carg = Var(model.ch_u, model.T, within=Binary)                                
    # VARIABLE Descarga efectiva de baterías (1 = Descarga) (0 = load_el/stand-by)
    model.Y_B_desc = Var(model.ch_u, model.T, within=Binary) 
    # VARIABLE Potencia de las baterías dirigida a la carga [kW]         
    model.PEL_B_L = Var(model.ch_u, model.bat_u, model.T, domain=NonNegativeReals)
    # VARIABLE Potencia de las baterías dirigida a los enfriadores eléctricos [kW]         
    model.PEL_B_EC = Var(model.ch_u, model.bat_u, model.T, domain=NonNegativeReals)
    # VARIABLE Potencia de las baterías dirigida a los enfriadores eléctricos [kW]         
    model.PEL_B_EH = Var(model.ch_u, model.bat_u, model.T, domain=NonNegativeReals)
    # VARIABLE Potencia de la red eléctrica a las bateria                 
    model.PEL_G_B = Var(model.ch_u, model.bat_u, model.T, domain=NonNegativeReals)  
    # VARIABLE Estado de carga de las baterías [kWh]                   
    model.SOC_B = Var(model.bat_u, model.T, domain=NonNegativeReals)   
    # VARIABLE Capacidad de las baterías tipo [kWh]                    
    model.CAP_B = Var(model.bat_u, model.T, domain=NonNegativeReals)   
    
    if data_model["batteries"]["active"] and data_model["inverters"]["active"]:

        # RESTRICCIÓN Número de strings baterías por tipo de batería e inversor
        def Batt_string_rule(m,tb,tch):
            return m.X_Bs[tb,tch] <= m.big_M*sum(m.X_CH[tpv,tb,tch] for tpv in model.pv_u)
        model.Batt_string_rule=Constraint(model.bat_u, model.ch_u,rule=Batt_string_rule)
        
        # RESTRICCIÓN Número de baterías por tipo de batería e inversor
        def Batt_num_rule(m, tb, tch):
            return m.X_B[tb,tch] == m.X_Bs[tb,tch]*np.floor(m.ch_f['V_n_batt',tch]/m.bat_f['V_nom',tb])
        model.Batt_num_rule=Constraint(model.bat_u, model.ch_u,rule=Batt_num_rule)

        # RESTRICCIÓN Cálculo de estado de carga en baterías cada hora
        def Batt_ts_rule(m,tb,t):#
            if t > m.T.first():
                return (m.SOC_B[tb,t] == m.SOC_B[tb,t-1]*(1-m.bat_f['Auto_des',tb]) + sum(m.bat_f['n',tb]*m.delta_t*(m.PEL_PV_B[tch,tb,t] + m.ch_f['n_acdc',tch]*(m.PEL_D_B[tch,tb,t] + m.PEL_CHP_B[tch,tb,t] + m.PEL_G_B[tch,tb,t] + m.PEL_WT_B[tch,tb,t])) -
                        (m.PEL_B_L[tch,tb,t] + m.PEL_B_EC[tch,tb,t] + m.PEL_B_EH[tch,tb,t])/(m.bat_f['n',tb]*m.ch_f['n_dcac',tch]) for tch in m.ch_u))
            else:
                return m.SOC_B[tb,t] == m.bat_f['Cap_nom',tb]*sum(m.X_B[tb,tch] for tch in m.ch_u)
        model.Batt_ts=Constraint(model.bat_u, model.T,rule=Batt_ts_rule)

        # RESTRICCIÓN Capacidad mínima de baterías 
        def Batt_socmin_rule(m,tb,t):#
            return m.SOC_B[tb,t] >= sum(m.X_B[tb,tch] for tch in m.ch_u)*m.bat_f['Cap_inf',tb]
        model.Batt_socmin=Constraint(model.bat_u, model.T,rule=Batt_socmin_rule)

        # RESTRICCIÓN Capacidad máxima de baterías 
        def Batt_socmax_rule(m,tb,t):#
            return m.SOC_B[tb,t] <= m.CAP_B[tb,t] 
        model.Batt_socmax=Constraint(model.bat_u, model.T,rule=Batt_socmax_rule)

        # RESTRICCIÓN Degradación de la capacidad de las baterías
        def Bcap_rule1(m,tb,t):#
            if t > m.T.first():
                return m.CAP_B[tb,t] == m.CAP_B[tb,t-1] - (m.bat_f['Deg_kwh',tb]/m.bat_f['n',tb])*sum((m.PEL_B_L[tch,tb,t] + m.PEL_B_EC[tch,tb,t] + + m.PEL_B_EH[tch,tb,t])/m.ch_f['n_dcac',tch] for tch in m.ch_u)
            else:
                return m.CAP_B[tb,t] == m.bat_f['Cap_nom',tb]*sum(m.X_B[tb,tch] for tch in m.ch_u)
        model.Bcap_rule1=Constraint(model.bat_u, model.T,rule=Bcap_rule1)

        # RESTRICCIÓN Degradación anual de la capacidad de las baterías
        def Bcap_rule2(m,tb):#
            return m.CAP_B[tb,m.T.first()]-m.CAP_B[tb,m.T.last()] <= 0.2*m.CAP_B[tb,m.T.first()]/m.bat_f['ty',tb]
        model.Bcap_rule2=Constraint(model.bat_u,rule=Bcap_rule2)

        # RESTRICCIÓN Potencia máxima de carga de las baterías límite técnico
        def PpvB_lim_rule1(m,tch,tb,t):
            return m.PEL_PV_B[tch,tb,t] + m.PEL_WT_B[tch,tb,t] + m.PEL_G_B[tch,tb,t] + m.PEL_D_B[tch,tb,t] + m.PEL_CHP_B[tch,tb,t] <= m.X_B[tb, tch]*m.bat_f['P_ch',tb]
        model.PpvB_lim_rule1=Constraint(model.ch_u, model.bat_u, model.T,rule=PpvB_lim_rule1)
        
        # RESTRICCIÓN Potencia máxima de carga de las baterías límite de los inversores
        def PpvB_lim_rule3(m,tch,tb,t):
            return m.PEL_PV_B[tch,tb,t] + m.PEL_WT_B[tch,tb,t] + m.PEL_G_B[tch,tb,t] + m.PEL_D_B[tch,tb,t] + m.PEL_CHP_B[tch,tb,t] <= (m.ch_f['V_n_batt',tch]*m.ch_f['I_max_ch_pv',tch]/1000)*sum(m.X_CH[tpv,tb,tch] for tpv in m.pv_u)
        model.PpvB_lim_rule3=Constraint(model.ch_u,model.bat_u,model.T,rule=PpvB_lim_rule3)

        # RESTRICCIÓN Potencia máxima de descarga de las baterías límite técnico
        def PBL_lims_rule1(m,tch,tb,t):
            return m.PEL_B_L[tch,tb,t] + m.PEL_B_EC[tch,tb,t] + m.PEL_B_EH[tch,tb,t] <= m.bat_f['P_des',tb]*m.X_B[tb, tch]
        model.PBL_lims_rule1=Constraint(model.ch_u, model.bat_u, model.T,rule=PBL_lims_rule1)

        # RESTRICCIÓN Potencia máxima de descarga de las baterías límite de los inversores
        def PBL_lims_rule2(m,tch,tb,t):
            return m.PEL_B_L[tch,tb,t] + m.PEL_B_EC[tch,tb,t] + m.PEL_B_EH[tch,tb,t] <= (m.ch_f['V_n_batt',tch]*m.ch_f['I_max_des',tch]/1000)*sum(m.X_CH[tpv,tb,tch] for tpv in m.pv_u)
        model.PBL_lims_rule2=Constraint(model.ch_u, model.bat_u, model.T,rule=PBL_lims_rule2)

        # RESTRICCIÓN Potencia máxima de carga de las baterías límite por binaria
        def Ceff_rule1(m,tch,t):#
            return sum(m.PEL_PV_B[tch,tb,t] + m.PEL_WT_B[tch,tb,t] + m.PEL_G_B[tch,tb,t] + m.PEL_D_B[tch,tb,t] + m.PEL_CHP_B[tch,tb,t] for tb in m.bat_u) <= m.big_M*m.Y_B_carg[tch,t]
        model.Ceff_rule1=Constraint(model.ch_u,model.T,rule=Ceff_rule1)

        # RESTRICCIÓN Potencia máxima de descarga de las baterías límite por binaria
        def Deff_rule1(m,tch,t):#
            return sum(m.PEL_B_L[tch,tb,t] + m.PEL_B_EC[tch,tb,t] + m.PEL_B_EH[tch,tb,t] for tb in m.bat_u) <= m.big_M*m.Y_B_desc[tch,t]
        model.Deff_rule1=Constraint(model.ch_u,model.T,rule=Deff_rule1)

        # RESTRICCIÓN Estado único de batería
        def Bstate_rule(model,tch,t):#
            return model.Y_B_carg[tch,t] + model.Y_B_desc[tch,t] <= 1
        model.Bstate_rule=Constraint(model.ch_u,model.T,rule=Bstate_rule)
    
    else:
        # RESTRICCIÓN-NULA Variables enteras y binarias baterías
        def Bat_null_num(m,tb,tch):
            return m.X_Bs[tb,tch] + m.X_B[tb,tch] == 0
        model.Bat_null_num=Constraint(model.bat_u, model.ch_u, rule=Bat_null_num)

        # RESTRICCIÓN-NULA Variables reales baterías
        def Bat_null_din(m,tch,t):
            return (
                m.Y_B_carg[tch,t] +  m.Y_B_desc[tch,t] + sum(m.PEL_B_L[tch,tb,t] for tb in m.bat_u) + 
                sum(m.PEL_B_EC[tch,tb,t] + m.PEL_B_EH[tch,tb,t] for tb in m.bat_u) + sum(m.PEL_G_B[tch,tb,t] for tb in m.bat_u)
            == 0)
        model.Bat_null_din = Constraint(model.ch_u, model.T, rule = Bat_null_din)

        # RESTRICCIÓN-NULA Variables de capacidad baterías
        def Bat_null_cap(m,tb,t):
            return m.SOC_B[tb,t] + m.CAP_B[tb,t] == 0
        model.Bat_null_cap = Constraint(model.bat_u, model.T, rule = Bat_null_cap)


    """
    -----------------------------------------------------------------------------------------
    ----------RESTRICCIONES DE INVERSORES HÍBRIDOS----------
    -----------------------------------------------------------------------------------------
    """
    
    if data_model["inverters"]["active"] and (data_model["batteries"]["active"] or data_model["pv_modules"]["active"]):

        if data_model["inverters"]["flex"]:
            # RESTRICCIÓN Se puede escoger más de una tecnología de inversor
            def PV_BATT_onetype_CH(m,tch):
                return sum(m.Y_CH[tpv,tb,tch] for tpv in m.pv_u for tb in m.bat_u) == 1
            model.PV_onetype_CH=Constraint(model.ch_u, rule=PV_BATT_onetype_CH)       
        else:
            # RESTRICCIÓN Se puede escoger solo una tecnología de inversor
            def PV_BATT_onetype_CH(m):
                return sum(m.Y_CH[tpv,tb,tch] for tpv in m.pv_u for tb in m.bat_u for tch in m.ch_u) == 1
            model.PV_onetype_CH=Constraint(rule=PV_BATT_onetype_CH)

        # RESTRICCIÓN Límite de binaria por método big_M
        def Bxch_rule(m,tpv,tb,tch):#
            return m.X_CH[tpv,tb,tch] <= m.big_M*m.Y_CH[tpv,tb,tch]
        model.Bxch_rule=Constraint(model.pv_u, model.bat_u, model.ch_u,rule=Bxch_rule)     

        # RESTRICCIÓN salida AC del inversor
        def ac_out_ch(m,tch,t):
            return m.PEL_PV_L[tch,t] + m.PEL_PV_EC[tch,t] + m.PEL_PV_EH[tch,t] + sum(m.PEL_B_L[tch,tb,t] + m.PEL_B_EC[tch,tb,t] + m.PEL_B_EH[tch,tb,t] for tb in m.bat_u) + m.PEL_PV_G[tch,t] <= m.ch_f['Pac_max_out',tch]*sum(m.X_CH[tpv,tb,tch] for tb in m.bat_u for tpv in m.pv_u)
        model.ac_out_ch = Constraint(model.ch_u, model.T,rule=ac_out_ch)

        # RESTRICCIÓN entrada AC del inversor
        def ac_in_ch(m,tch,tb,t):
            return m.PEL_G_B[tch,tb,t] + m.PEL_WT_B[tch,tb,t] + m.PEL_D_B[tch,tb,t] + m.PEL_CHP_B[tch,tb,t] <= m.ch_f['Pac_max_in',tch]*sum(m.X_CH[tpv,tb,tch] for tpv in m.pv_u)
        model.ac_in_ch = Constraint(model.ch_u, model.bat_u,model.T,rule=ac_in_ch)                   

    else:
        # RESTRICCIÓN-NULA Variables de enteras y binarias inversores
        def Inv_null(m,tpv,tb,tch):#
            return m.X_CH[tpv,tb,tch] + m.Y_CH[tpv,tb,tch] == 0
        model.Inv_null = Constraint(model.pv_u, model.bat_u, model.ch_u, rule = Inv_null)            

    """
    -----------------------------------------------------------------------------------------
            ----------PARÁMETROS Y VARIABLES DE LOS CALENTADORES ELÉCTRICOS----------
    -----------------------------------------------------------------------------------------
    """
    if data_model["el_heaters"]["active"]:
        model.eh_u = Set(initialize=data_model["el_heaters"]["type"].columns.tolist())
        model.eh_f = Param(data_model["el_heaters"]["type"].index.to_list(), model.eh_u, initialize = create_dict(data_model["el_heaters"]["type"]), domain = Any)
    else:
        # SET-NULO Tecnologías de calentadores eléctricos
        model.eh_u = Set(initialize=["None"])

        # PARÁMETRO-NULO Características tecnologías de calentadores eléctricos
        EH_none_df = pd.DataFrame(index=["C_inst", "C_OM_kWh"], data={"None":[0,0]})
        model.eh_f = Param(EH_none_df.index.to_list(), model.eh_u, initialize = create_dict(EH_none_df), domain = Any)    

    # VARIABLE Número de calentadores de absorción
    model.X_EH = Var(model.eh_u, domain=NonNegativeIntegers)  
    # VARIABLE Binaria, el enfriador de absorción está en funcionamiento
    model.Y_EH = Var(model.T, model.eh_u, within=Binary)     
    # VARIABLE Potencia de calor generada por los calentadores eléctricos [kWcl]
    model.PTH_EH = Var(model.T, model.eh_u,domain=NonNegativeReals)    
    # VARIABLE Potencia eléctrica consumida por los calentadores eléctricos [kWth]
    model.PEL_EH = Var(model.T, model.eh_u, domain=NonNegativeReals)
    # VARIABLE Potencia de la red eléctrica a los calentadores eléctricos                           
    model.PEL_G_EH = Var(model.T, domain=NonNegativeReals)  
    # VARIABLE Potencia térmica de los calentadores eléctricos a la carga                          
    model.PTH_EH_L = Var(model.T, domain=NonNegativeReals)  
        
    if data_model["el_heaters"]["active"]:        
        
        # RESTRICCIÓN Límite superior de potencia de salida del calentador eléctrico
        def max_p_eh(m,t,teh):
            return m.PTH_EH[t,teh] <= m.eh_f['P_nom',teh]*m.X_EH[teh]
        model.max_p_eh = Constraint(model.T, model.eh_u, rule=max_p_eh)

        # RESTRICCIÓN Límite inferior de potencia de salida del calentador eléctrico
        def min_p_eh(m,t,teh):
            return m.PTH_EH[t,teh] + m.big_M*(1-m.Y_EH[t,teh]) >= m.eh_f['P_min_porc',teh]*m.eh_f['P_nom',teh]*m.X_EH[teh] 
        model.min_p_eh = Constraint(model.T, model.eh_u, rule=min_p_eh)

        # RESTRICCIÓN Conversión a energía de calor considerando eficiencia
        def efficiency_eh(m,t,teh):
            return m.PEL_EH[t,teh]*m.eh_f['n',teh] == m.PTH_EH[t,teh] 
        model.efficiency_eh = Constraint(model.T, model.eh_u, rule=efficiency_eh)

        # RESTRICCIÓN Balance de potencia eléctrica que consumen los calentadores eléctricos
        def PEL_balance_eh(m,t):
            return (
                sum(m.PEL_EH[t,teh] for teh in m.eh_u) == 
                sum(sum(m.PEL_B_EH[tch,tb,t] for tb in m.bat_u) + # De las baterías
                m.ch_f['n_dcac',tch]*m.PEL_PV_EH[tch,t] for tch in m.ch_u) + # De los paneles solares
                m.PEL_G_EH[t] + # De la red
                m.PEL_WT_EH[t] + # De las turbinas eólicas
                m.PEL_D_EH[t] # Del diesel
            )
        model.PEL_balance_eh = Constraint(model.T,rule=PEL_balance_eh)

        # RESTRICCIÓN Balance de potencia térmica de los calentadores eléctricos 
        def PTH_balance_eh(m,t):
            return (
                sum(m.PTH_EH[t,teh] for teh in m.eh_u) == 
                sum(m.PTH_EH_AC[t,tac] for tac in m.ac_u) + # Al enfriador de absorbsión
                m.PTH_EH_L[t] # A la carga
            )
        model.PTH_balance_eh = Constraint(model.T,rule=PTH_balance_eh)

    else:       

        # RESTRICCIÓN NULA Variables binarias enfriadores eléctricos
        def None_bin_eh(m,teh):
            return sum(m.Y_EH[t,teh] for t in m.T) + model.X_EH[teh] == 0
        model.None_bin_eh = Constraint(model.eh_u, rule=None_bin_eh)
        
        # RESTRICCIÓN NULA Funcionamiento enfriadores eléctricos
        def None_fun_eh(m,t,teh):
            return m.PTH_EH[t,teh] + m.PEL_EH[t,teh] + m.PTH_EH_L[t] == 0
        model.None_fun_eh = Constraint(model.T, model.eh_u, rule=None_fun_eh)


    """
    -----------------------------------------------------------------------------------------
            ----------PARÁMETROS Y VARIABLES DE LOS ENFRIADORES ELÉCTRICOS----------
    -----------------------------------------------------------------------------------------
    """

    if data_model["el_chillers"]["active"]:
        # SET Tecnologías de enfriadores eléctricos
        model.ec_u = Set(initialize=data_model["el_chillers"]["type"].columns.tolist())
        # PARÁMETRO Características tecnologías de enfriadores eléctricos
        model.ec_f = Param(data_model["el_chillers"]["type"].index.to_list(), model.ec_u, initialize = create_dict(data_model["el_chillers"]["type"]), domain = Any)
    else:
        # SET NULO Tecnologías de enfriadores eléctricos
        model.ec_u = Set(initialize=["None"])

        # PARÁMETRO NULO Características tecnologías de enfriadores eléctricos
        EC_none_df = pd.DataFrame(index=["C_inst", "C_OM_kWh"], data={"None":[0,0]})
        model.ec_f = Param(EC_none_df.index.to_list(), model.ec_u, initialize = create_dict(EC_none_df), domain = Any)


    # VARIABLE Número de enfriadores de absorción
    model.X_EC = Var(model.ec_u, domain=NonNegativeIntegers)  
    # VARIABLE Binaria, el enfriador de absorción está en funcionamiento
    model.Y_EC = Var(model.T, model.ec_u, within=Binary)     
    # VARIABLE Potencia de enfriamiento generada por los enfriadores de absorción [kWcl]
    model.PCL_EC = Var(model.T, model.ec_u,domain=NonNegativeReals)
    # VARIABLE Auxiliar potencia nominal enfriadores de absorción (restricciones Eficiencia de carga parcial) [kWcl]
    model.PNOM_EC_AUX = Var(model.T, model.ec_u,domain=NonNegativeReals)
    # VARIABLE Potencia eléctrica consumida por los enfriadores eléctricos [kWth]
    model.PEL_EC = Var(model.T, model.ec_u, domain=NonNegativeReals)
    # VARIABLE Potencia de la red eléctrica a los enfriadores eléctricos                           
    model.PEL_G_EC = Var(model.T, domain=NonNegativeReals)  
        
    if data_model["el_chillers"]["active"]:        
        
        # RESTRICCIÓN Límite superior de potencia de salida del enfriador de absorción
        def max_p_ec(m,t,tec):
            return m.PCL_EC[t,tec] <= m.ec_f['P_cl_nom',tec]*m.X_EC[tec]
        model.max_p_ec = Constraint(model.T, model.ec_u, rule=max_p_ec)

        # RESTRICCIÓN Límite inferior de potencia de salida del enfriador de absorción
        def min_p_ec(m,t,tec):
            return m.PCL_EC[t,tec] + m.big_M*(1-m.Y_EC[t,tec]) >= m.ec_f['P_min_porc',tec]*m.ec_f['P_cl_nom',tec]*m.X_EC[tec] 
        model.min_p_ec = Constraint(model.T, model.ec_u, rule=min_p_ec)

        # RESTRICCIÓN Conversión a energía de enfriamiento considerando eficiencia de carga parcial
        def efficiency_ec(m,t,tec):
            return m.PEL_EC[t,tec] == m.ec_f['y_n',tec]*m.PNOM_EC_AUX[t,tec] + m.ec_f['lamd_n',tec]*m.PCL_EC[t,tec] 
        model.efficiency_ec = Constraint(model.T, model.ec_u, rule=efficiency_ec)

        # RESTRICCIÓN Restricción 1 potencia nominal auxiliar enfriadores de absorción
        def ec_nom_aux_1(m,t,tec):
            return m.PNOM_EC_AUX[t,tec] <= m.big_M*m.Y_EC[t,tec]
        model.ec_nom_aux_1 = Constraint(model.T, model.ec_u, rule=ec_nom_aux_1)

        # RESTRICCIÓN Restricción 2 potencia nominal auxiliar enfriadores de absorción
        def ec_nom_aux_2(m,t,tec):
            return m.PNOM_EC_AUX[t,tec] <= m.ec_f['P_cl_nom',tec]*m.X_EC[tec] 
        model.ec_nom_aux_2 = Constraint(model.T, model.ec_u, rule=ec_nom_aux_2)

        # RESTRICCIÓN Restricción 3 potencia nominal auxiliar enfriadores de absorción
        def ec_nom_aux_3(m,t,tec):
            return m.PNOM_EC_AUX[t,tec] >= m.ec_f['P_cl_nom',tec]*m.X_EC[tec] - m.big_M*(1-m.Y_EC[t,tec])
        model.ec_nom_aux_3 = Constraint(model.T, model.ec_u, rule=ec_nom_aux_3)

        # RESTRICCIÓN Balance de potencia eléctrica que consumen los enfriadores eléctricos
        def PEL_balance_ec(m,t):
            return (
                sum(m.PEL_EC[t,tec] for tec in m.ec_u) == 
                sum(sum(m.PEL_B_EC[tch,tb,t] for tb in m.bat_u) + 
                m.ch_f['n_dcac',tch]*m.PEL_PV_EC[tch,t] for tch in m.ch_u) + m.PEL_G_EC[t] + 
                m.PEL_WT_EC[t] + m.PEL_D_EC[t]
            )
        model.PEL_balance_ec = Constraint(model.T,rule=PEL_balance_ec) 

    else:       

        # RESTRICCIÓN NULA Variables binarias enfriadores eléctricos
        def None_bin_ec(m,tec):
            return sum(m.Y_EC[t,tec] for t in m.T) + model.X_EC[tec] == 0
        model.None_bin_ec = Constraint(model.ec_u, rule=None_bin_ec)
        
        # RESTRICCIÓN NULA Funcionamiento enfriadores eléctricos
        def None_fun_ec(m,t,tec):
            return m.PCL_EC[t,tec] + m.PEL_EC[t,tec] + m.PNOM_EC_AUX[t,tec] == 0
        model.None_fun_ec = Constraint(model.T, model.ec_u, rule=None_fun_ec)
    

    """
    -----------------------------------------------------------------------------------------
            ----------PARÁMETROS Y VARIABLES RED ELÉCTRICA----------
    -----------------------------------------------------------------------------------------
    """   
    # VARIABLE Potencia de la red eléctrica a la carga eléctrica
    model.PEL_G_L = Var(model.T, domain=NonNegativeReals)   

    if data_model["grid_el"]["active"]:

        model.max_p_grid_el_buy = Param(initialize = data_model["grid_el"]["pmax_buy"])   # PARÁMETRO Potencia máxima de compra de la red eléctrica
        model.max_p_grid_el_sell = Param(initialize = data_model["grid_el"]["pmax_sell"]) # PARÁMETRO Potencia máxima de venta a la red eléctrica

        # PARÁMETRO Disponibilidad de la red eléctrica
        if data_model["grid_el"]["av"]["active"]:
            model.grid_el_av = Param(model.T, initialize = T_dict(T, data_model["grid_el"]["av"]["value"])) 
        else:
            model.grid_el_av = Param(model.T, initialize = T_dict(T, np.repeat(1, len(T)))) 

        # RESTRICCIÓN Balance y potencia límite de compra red eléctrica
        def PG_lim_rule(m,t):
            return m.PEL_G_L[t] + m.PEL_G_EC[t] + m.PEL_G_EH[t] + sum(m.PEL_G_B[tch,tb,t] for tb in m.bat_u for tch in m.ch_u) <= m.grid_el_av[t]*m.max_p_grid_el_buy
        model.PG_lim=Constraint(model.T,rule=PG_lim_rule)
        
        # RESTRICCIÓN Balance y potencia límite de venta red eléctrica
        def PpvG_lim_rule(m,t):
            return sum(m.PEL_PV_G[tch,t] for tch in m.ch_u) + m.PEL_WT_G[t] + m.PEL_CHP_G[t] <= m.grid_el_av[t]*m.max_p_grid_el_sell
        model.PpvG_lim=Constraint(model.T,rule=PpvG_lim_rule)

        # PARÁMETRO Precio de compra de energía de la red eléctrica
        if data_model["grid_el"]["buy_price"]["type"] == "fixed":
            model.price_buy_grid_el = Param(model.T, initialize = T_dict(T, np.repeat(data_model["grid_el"]["buy_price"]["value"], len(T)))) 
        elif data_model["grid_el"]["buy_price"]["type"] == "variable":
            model.price_buy_grid_el = Param(model.T, initialize = T_dict(T, data_model["grid_el"]["buy_price"]["value"])) 
        
        # PARÁMETRO Precio de venta de energía de la red eléctrica
        if data_model["grid_el"]["sell_price"]["type"] == "fixed":
            model.price_sell_grid_el = Param(model.T, initialize= T_dict(T, np.repeat(data_model["grid_el"]["sell_price"]["value"], len(T)))) 
        elif data_model["grid_el"]["sell_price"]["type"] == "variable":
            model.price_sell_grid_el  = Param(model.T, initialize = T_dict(T, data_model["grid_el"]["sell_price"]["value"])) 

    else:        
        # PARÁMETRO NULO Precio de compra de energía de la red eléctrica
        model.price_buy_grid_el = Param(model.T, initialize = T_dict(T, np.repeat(0, len(T))))     
        # PARÁMETRO NULO Precio de venta de energía de la red eléctrica
        model.price_sell_grid_el = Param(model.T, initialize= T_dict(T, np.repeat(0, len(T))))   
        # PARÁMETRO NULO Disponibilidad de la red eléctrica
        model.grid_el_av = Param(model.T, initialize= T_dict(T, np.repeat(0, len(T))))   

        # RESTRICCIÓN NULA Balance variables de la red eléctrica
        def None_grid(m,t):
            return (
                m.PEL_G_L[t] + m.PEL_G_EC[t] + m.PEL_G_EH[t] 
                + sum(m.PEL_G_B[tch,tb,t] for tb in m.bat_u for tch in m.ch_u) 
                + sum(m.PEL_PV_G[tch,t] for tch in m.ch_u) + m.PEL_WT_G[t] 
                + m.PEL_CHP_G[t]
                == 0
            )
        model.None_grid=Constraint(model.T,rule=None_grid)



    """
    -----------------------------------------------------------------------------------------
            ----------PARÁMETROS Y VARIABLES RED DE GAS----------
    -----------------------------------------------------------------------------------------
    """   
    if data_model["grid_gas"]["active"]:

        # PARÁMETRO Potencia máxima de compra de la red de gas
        model.max_p_grid_gas_buy = Param(initialize = data_model["grid_gas"]["pmax_buy"])   
        
        # PARÁMETRO Disponibilidad de la red de gas
        if data_model["grid_gas"]["av"]["active"]:
            model.grid_gas_av = Param(model.T, initialize = T_dict(T, data_model["grid_gas"]["av"]["value"])) 
        else:
            model.grid_gas_av = Param(model.T, initialize = T_dict(T, np.repeat(1, len(T)))) 

        # RESTRICCIÓN Balance y potencia límite de compra red de gas
        def red_gas_lim(m,t):
            return sum(m.PPE_GAS_BOI[t,tboi] for tboi in m.boi_u) + sum(m.PPE_GAS_CHP[t,tchp] for tchp in m.chp_u)  <= m.grid_el_av[t]*m.max_p_grid_el_buy
        model.red_gas_lim = Constraint(model.T, rule = red_gas_lim)
        
        # PARÁMETRO Precio de compra de energía de la red de gas
        if data_model["grid_gas"]["buy_price"]["type"] == "fixed":
            model.price_buy_grid_gas = Param(model.T, initialize = T_dict(T, np.repeat(data_model["grid_gas"]["buy_price"]["value"], len(T)))) 
        elif data_model["grid_gas"]["buy_price"]["type"] == "variable":
            model.price_buy_grid_gas = Param(model.T, initialize = T_dict(T, data_model["grid_gas"]["buy_price"]["value"])) 
     
    else:        
        # PARÁMETRO NULO Precio de compra de energía de la red de gas
        model.price_buy_grid_gas = Param(model.T, initialize = T_dict(T, np.repeat(0, len(T))))          
        # PARÁMETRO NULO Disponibilidad de la red eléctrica
        model.grid_gas_av = Param(model.T, initialize= T_dict(T, np.repeat(0, len(T))))   

        # RESTRICCIÓN NULA Balance variables de la red de gas
        def None_grid_gas(m,t):
            return sum(m.PPE_GAS_BOI[t,tboi] for tboi in m.boi_u) + sum(m.PPE_GAS_CHP[t,tchp] for tchp in m.chp_u) == 0
        model.None_grid_gas=Constraint(model.T,rule=None_grid_gas)

    # RESTRICCIÓN Balance de Potencia de la carga eléctrica
    def PEL_balance_load(m,t):
        return (
            m.PEL_G_L[t]+ sum(m.PEL_B_L[tch,tb,t] for tb in m.bat_u for tch in m.ch_u) 
            + sum(m.ch_f['n_dcac',tch]*m.PEL_PV_L[tch,t] for tch in m.ch_u) + m.PEL_D_L[t] 
            + m.PEL_CHP_L[t] + m.PEL_WT_L[t] + m.PEL_NS[t] 
            == m.load_el[t] + sum(m.L_NEEDS[t,nds,"Electrico"] for nds in model.needs)
        )
    model.PEL_balance_load=Constraint(model.T,rule=PEL_balance_load)

    # RESTRICCIÓN Balance de Potencia de la carga térmica
    def PTH_balance_load(m,t):
        return (
            m.PTH_BOI_L[t] # De las calderas
            + m.PTH_EH_L[t] # De los calentadores eléctricos
            + m.PTH_CHP_L[t] # De los CHP
            == 
            m.load_th[t] + sum(m.L_NEEDS[t,nds,"Calor"] for nds in model.needs)
        )
    model.PTH_balance_load=Constraint(model.T,rule=PTH_balance_load)

    # RESTRICCIÓN Balance de Potencia de la carga de refrigeración
    def PCL_balance_load(m,t):
        return (
            sum(m.PCL_AC[t,tac] for tac in m.ac_u) 
            + sum(m.PCL_EC[t,tec] for tec in m.ec_u) 
            #+ m.PCL_NS[t] 
            == 
            m.load_cl[t] + sum(m.L_NEEDS[t,nds,"Refrigeracion"] for nds in model.needs)
        )
    model.PCL_balance_load=Constraint(model.T,rule=PCL_balance_load)    
  
    
    if data_model["area"]["active"]:
        model.Area = Param(initialize = data_model["area"]["value"])
        # RESTRICCIÓN Área de instalación para paneles
        def PV_number_rule(m):#
            return sum(m.X_PV[tpv,tch]*m.pv_f['A',tpv]  for tch in m.ch_u for tpv in m.pv_u) + sum(m.X_WT[tt]*m.wt_f['A',tt] for tt in m.wt_u) <= m.Area
        model.PV_number=Constraint(rule=PV_number_rule)


    #if data_model["max_invest"]["active"]:
    #    model.MaxBudget = Param(initialize = data_model["max_invest"]["value"])
    #    # RESTRICCIÓN Máxima inversión        
    #    def Budget_rule(m):#
    #        return sum(m.X_PV[tpv,tch]*(m.pv_f['C_inst',tpv]) for tch in m.ch_u for tpv in m.pv_u) + sum(m.X_B[tb,tch]*m.bat_f['C_inst',tb] for tch in m.ch_u for tb in m.bat_u) \
    #               + sum(m.X_CH[tpv,tb,tch]*m.ch_f['C_inst',tch] for tpv in m.pv_u for tb in m.bat_u for tch in m.ch_u) + sum(m.X_WT[tt]*m.wt_f['C_inst',tt] for tt in m.wt_u) + model.d_cost_inst <= m.MaxBudget
    #    model.Budget=Constraint(rule=Budget_rule)

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
            + sum(m.X_EH[teh]*m.eh_f['C_inst',teh] for teh in m.eh_u)
            + sum(m.X_CHP[tchp]*m.chp_f['C_inst',tchp] for tchp in m.chp_u)
            + sum(m.X_AC[tac]*m.ac_f['C_inst',tac] for tac in m.ac_u)
            + sum(m.X_EC[tec]*m.ec_f['C_inst',tec] for tec in m.ec_u)
            
            + sum(sum(VPN_F[ii-1]*m.ch_f['C_inst',tch]*sum(m.X_CH[tpv,tb,tch] for tb in m.bat_u for tpv in m.pv_u) for ii in np.arange(int(m.ch_f['ty',tch]),m.lifeyears,int(m.ch_f['ty',tch]))) for tch in m.ch_u)
            + sum(sum(VPN_F[ii-1]*m.bat_f['C_inst',tb]*sum(m.X_B[tb,tch] for tch in m.ch_u) for ii in np.arange(int(m.bat_f['ty',tb]),m.lifeyears,int(m.bat_f['ty',tb]))) for tb in m.bat_u)
            + sum(sum(VPN_F[ii-1]*m.wt_f['C_inst',tt]*m.X_WT[tt] for ii in np.arange(int(m.wt_f['ty',tt]),m.lifeyears,int(m.wt_f['ty',tt]))) for tt in m.wt_u)
            
            + sum(sum(VPN_F[ii-1]*m.boi_f['C_inst',tboi]*m.X_BOI[tboi] for ii in np.arange(int(m.boi_f['ty',tboi]),m.lifeyears,int(m.boi_f['ty',tboi]))) for tboi in m.boi_u)
            + sum(sum(VPN_F[ii-1]*m.eh_f['C_inst',teh]*m.X_EH[teh] for ii in np.arange(int(m.eh_f['ty',teh]),m.lifeyears,int(m.eh_f['ty',teh]))) for teh in m.eh_u)
            + sum(sum(VPN_F[ii-1]*m.chp_f['C_inst',tchp]*m.X_CHP[tchp] for ii in np.arange(int(m.chp_f['ty',tchp]),m.lifeyears,int(m.chp_f['ty',tchp]))) for tchp in m.chp_u)
            + sum(sum(VPN_F[ii-1]*m.ac_f['C_inst',tac]*m.X_AC[tac] for ii in np.arange(int(m.ac_f['ty',tac]),m.lifeyears,int(m.ac_f['ty',tac]))) for tac in m.ac_u)
            + sum(sum(VPN_F[ii-1]*m.ec_f['C_inst',tec]*m.X_EC[tec] for ii in np.arange(int(m.ec_f['ty',tec]),m.lifeyears,int(m.ec_f['ty',tec]))) for tec in m.ec_u)
        )
    model.FirstStage = Expression(rule=ComputeFirstStageCost)

    def ComputeSecondStageCost(m):
        return VPN_FS*sum(
            m.price_buy_grid_el[t]*(m.PEL_G_L[t] + m.PEL_G_EC[t] + m.PEL_G_EH[t] + sum(m.PEL_G_B[tch,tb,t] for tb in m.bat_u for tch in m.ch_u))
            
            + m.price_buy_grid_gas[t]*(sum(m.PPE_GAS_BOI[t,tboi] for tboi in m.boi_u) + sum(m.PPE_GAS_CHP[t,tchp] for tchp in m.chp_u))
            
            + m.d_fuel_cost*(m.d_f_min*m.Y_D[t] + m.d_fm*(m.PEL_D_L[t] + m.PEL_D_EC[t] + m.PEL_D_EH[t] + sum(m.PEL_D_B[tch,tb,t] for tch in m.ch_u for tb in m.bat_u))) 
            
            + m.d_cost_om*m.Y_D[t]
            + sum(m.PTH_BOI[t,tboi]*m.boi_f['C_OM_kWh',tboi] for tboi in m.boi_u)
            + sum(m.PTH_EH[t,teh]*m.eh_f['C_OM_kWh',teh] for teh in m.eh_u)
            + sum(m.PEL_CHP[t,tchp]*m.chp_f['C_OM_kWh',tchp] for tchp in m.chp_u)
            + sum(m.PCL_AC[t,tac]*m.ac_f['C_OM_kWh',tac] for tac in m.ac_u)
            + sum(m.PCL_EC[t,tec]*m.ec_f['C_OM_kWh',tec] for tec in m.ec_u)
            
            + m.cost_ens_el[t]*m.PEL_NS[t] #+ m.cost_ens_th[t]*m.PTH_NS[t] + m.cost_ens_cl[t]*m.PCL_NS[t]
            
            - m.price_sell_grid_el[t]*(sum(m.ch_f['n_dcac',tch]*m.PEL_PV_G[tch,t] for tch in m.ch_u) + m.PEL_WT_G[t] + m.PEL_CHP_G[t])
            - m.EnvC*sum(sum(m.X_PV[tpv,tch]*m.p_pv_gen[t,tpv] for tpv in m.pv_u) - m.PEL_PV_CUR[tch,t] for tch in m.ch_u)
            - m.EnvC*(sum(m.X_WT[tt]*m.p_wt_gen[t,tt] for tt in m.wt_u) - m.PEL_WT_CUR[t]) 
            
        for t in m.T)

    model.SecondStage = Expression(rule=ComputeSecondStageCost)
     
    def obj_rule(m):
        return  m.FirstStage + m.SecondStage
    model.Obj=Objective(rule=obj_rule,sense=minimize)                 

    return model
    

