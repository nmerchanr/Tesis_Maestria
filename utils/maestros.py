import os
import json
import re
import pandas as pd
import numpy as np
import pickle

import mpisppy.utils.sputils as sputils
from mpisppy.opt.lshaped import LShapedMethod

from utils.functions import power_PV_calculation, calculate_WT_power
from utils.model import create_model

from pyomo.environ import value
import pyomo.environ as pyo



class Maestro_Optimizacion():

    def __init__(self, data_model : dict, location : dict, df_meteo : pd.DataFrame,  info_meteo : pd.DataFrame, estocastico = False):
        self.location = location.copy()
        self.data_model = data_model.copy()
        self.df_meteo = df_meteo.copy()
        self.info_meteo = info_meteo.copy()
        self.estocastico = estocastico
       
        
        self.VPN_F = np.sum([round(1/np.power(1+self.data_model["interest"],i),3) for i in np.arange(1,self.data_model["lifeyears"]+1)])

        if self.estocastico:

            self.ruta_resultados = f'RESULTADOS/{self.location["name_data"]}/estocastico'

            self.GHI_scens = pd.read_csv(f'Escenarios/GHI_scens_{self.location["name_data"]}.csv')
            self.wind_scens = pd.read_csv(f'Escenarios/wind_scens_{self.location["name_data"]}.csv')

            with open(f'Escenarios/GHI_prob_{self.location["name_data"]}.json') as user_file:
                self.GHI_prob = json.loads(user_file.read())
            with open(f'Escenarios/wind_prob_{self.location["name_data"]}.json') as user_file:
                self.wind_prob = json.loads(user_file.read())

        else:

            self.ruta_resultados = f'RESULTADOS/{self.location["name_data"]}/determinista'

            self.GHI_scens = pd.DataFrame({"main" : df_meteo["GHI"].values})
            self.wind_scens = pd.DataFrame({"main" : df_meteo["Wind Speed"].values})

            self.GHI_prob = {"main" : 1}
            self.wind_prob = {"main" : 1}

        self.escenarios = [f'{key_irr}_{key_wind}' for key_irr in self.GHI_prob for key_wind in self.wind_prob]

    def scenario_creator(self, scenario_name : str):

        scens = scenario_name.split("_")

        df_scen = self.df_meteo.copy()

        df_scen["GHI"] = self.GHI_scens[scens[0]].to_numpy().astype(float)
        df_scen["Wind Speed"] = self.wind_scens[scens[1]].to_numpy().astype(float)

        if df_scen.isnull().values.any():
            raise ValueError("NaN detected")

        if self.data_model["pv_modules"]["active"]:
            self.data_model["pv_modules"]["Pmpp"] = power_PV_calculation(df_scen, self.data_model["pv_modules"]["type"], 0, 10, self.data_model["lat"], type="electric") 
        
        if self.data_model["pv_thermal"]["active"]:
            self.data_model["pv_thermal"]["Pmpp"] = power_PV_calculation(df_scen, self.data_model["pv_thermal"]["type"], 0, 10, self.data_model["lat"], type="thermal")   
        
        if self.data_model["windgen"]["active"]:
            _, Wind_generation = calculate_WT_power(df_scen, self.data_model["windgen"]["type"], 0.001, 20, self.info_meteo['Elevation'].iloc[0])               
            self.data_model["windgen"]["generation"] = Wind_generation
        
        model = create_model(self.data_model)

        vars_first_stage = [
            model.X_PV,
            model.X_PVs,
            model.X_PT,
            model.X_AT,
            model.X_B,
            model.X_Bs,
            model.X_CH,
            model.Y_CH,
            model.X_WT,
            model.X_BOI,
            model.X_EH,
            model.X_CHP,
            model.X_AC,
            model.X_EC,
            model.Y_NEEDS
        ]

        sputils.attach_root_node(model, model.FirstStage, vars_first_stage)
        model._mpisppy_probability = self.GHI_prob[scens[0]]*self.wind_prob[scens[1]]
        
        return model
    
    def scenario_tree_solution_writer(self, directory_name, scenario_name, scenario, bundling ):
        with open(os.path.join(directory_name, scenario_name+'.csv'), 'w', encoding= "utf-8") as f:
            for var in scenario.component_data_objects(
                    ctype=(pyo.Var, pyo.Expression),
                    descend_into=True,
                    active=True,
                    sort=True):
                var_name = var.name
                if bundling:
                    dot_index = var_name.find('.')
                    assert dot_index >= 0
                    var_name = var_name[(dot_index+1):]
                f.write(f"{var_name};{pyo.value(var)}\n")

    def guardar_resultados(self):

        os.makedirs(self.ruta_resultados, exist_ok=True)        
        self.ls.write_tree_solution(self.ruta_resultados, self.scenario_tree_solution_writer)

        print(f"Resultados guardados con exito en: {self.ruta_resultados}")

    def corregir_indices(self, cadena):
        # Expresión regular para encontrar los índices dentro de los corchetes
        patron = r"\[([^\]]+)\]"  # Captura lo que está dentro de los corchetes

        def reemplazo(match):
            # Obtener el contenido dentro de los corchetes
            contenido = match.group(1)
            
            # Usamos una expresión regular para separar por comas fuera de paréntesis
            indices = re.split(r",(?![^\(]*\))", contenido)  # Separar por coma pero no dentro de paréntesis
            
            # Procesar cada índice
            for i in range(len(indices)):
                # Eliminar espacios en blanco alrededor de los índices
                indices[i] = indices[i].strip()
                # Si el índice no es numérico y no está entre comillas, lo envuelve en comillas simples
                if not indices[i].isdigit() and not (indices[i].startswith("'") and indices[i].endswith("'")):
                    indices[i] = f"'{indices[i]}'"
            
            # Unir los índices corregidos con comas
            return f"[{','.join(indices)}]"

        # Aplicar la expresión regular para corregir los índices en la cadena
        return re.sub(patron, reemplazo, cadena)
    

    def pertenece_variable(self, modelo, variable_string):
        # Expresión regular para dividir el nombre y los índices
        match = re.match(r"(\w+)(?:\[(.+)\])?", variable_string)
        if not match:
            raise ValueError("Formato de la variable inválido. Debe ser 'NOMBRE' o 'NOMBRE[índices]'")
        
        variable_name = match.group(1)  # Nombre de la variable

        # Verificar si la variable existe en el modelo
        return variable_name in modelo.component_map(pyo.Var)   
    
    def leer_resultados(self):
        
        self.model_escenarios = {}

        for esc in self.escenarios:
            ruta_archivo = f'{self.ruta_resultados}/{esc}.csv'

            df = pd.read_csv(ruta_archivo, sep=";", header = None, encoding="utf-8")

            df.columns = ["variable","valor"]
            df.dropna(inplace= True)            
            
            model = self.scenario_creator(esc)

            for _, item in df.iterrows():
                
                variable = self.corregir_indices(item["variable"])

                if self.pertenece_variable(model, variable):
                                
                    cond_bin = eval(f'model.{variable}.domain is pyo.Binary') 
                                    
                    if cond_bin:
                        valor = round(item["valor"])
                    else:
                        valor = item["valor"]                

                    exec(f'model.{variable}.value = {valor}')

                else: 
                    print("Variable no encontrada", variable)                      

            self.model_escenarios[esc] = model

    def guardar_progreso(self, archivo="progreso_lshaped.pkl"):

        path = "backup_optimizacion" 
        os.makedirs(path, exist_ok=True)
        
        progreso = {
            "eta_bounds": self.ls.eta_bounds,
            "iteration_history": self.ls.iteration_history,
            "current_objective": self.ls.current_objective,
        }

        with open(f"{path}/{archivo}", "wb") as f:
            pickle.dump(progreso, f)
        print("Progreso guardado.")

    def cargar_progreso(self, archivo="progreso_lshaped.pkl"):
        
        path = "backup_optimizacion" 

        with open(f"{path}/{archivo}", "rb") as f:
            progreso = pickle.load(f)
        
        self.ls.eta_bounds = progreso["eta_bounds"]
        self.ls.iteration_history = progreso["iteration_history"]
        self.current_objective = progreso["current_objective"]

    def LS_optimizacion(self, max_iter : int = 500, bound = 0, cargar_progreso = False):  
        
        options = {
            "root_solver": "cplex",
            "sp_solver": "cplex",
            "sp_solver_options": {
                "threads": 16,  # Número óptimo de hilos (demasiados pueden ralentizar)
                "parallel": -1,  # CPLEX elige el mejor paralelismo
                "workmem": 16384,  # Asigna más memoria (8GB)
                "mip_tolerances_mipgap": 0.01,  # Relaja el criterio de optimalidad (0.5%)
                "mip_tolerances_absmipgap": 1.0,  # Margen absoluto de optimalidad
                "mip_strategy_rinsheur": 10,  # Refuerzo de heurística RINS
                "mip_cuts_mircut": 2,  # Activar cortes MIR
                "mip_cuts_gomory": 2,  # Activar cortes Gomory
                "mip_cuts_flowcovers": 2,  # Activar cortes de cobertura de flujo
                "mip_strategy_startalgorithm": 3,  # Comienza con el método de barrera
                "mip_strategy_bbinterval": 5,  # Ajusta la exploración de ramas en Branch & Bound
                "mip_display": 2,  # Muestra más información en la consola
            },
            "max_iter": max_iter,  # Limitar el número de iteraciones para no sobrecargar
            "valid_eta_lb": {name: bound for name in self.escenarios},
            "verbose": True
        }

        print(options)

        self.ls = LShapedMethod(options, self.escenarios, self.scenario_creator)

        if cargar_progreso:
            self.cargar_progreso()
        
        try:
            result = self.ls.lshaped_algorithm()
            self.guardar_resultados()
        except Exception as e:
            self.guardar_progreso()
            print(f"Error durante la ejecución: {e}")        


class Maestro_Resultados():

    def __init__(self):
        pass

    def resultado_valor_inversion_inicial(self):

        m = None    
        
        self.componentes_inversion_inicial = {
            "Paneles solares eléctricos" : {"index" : [m.pv_u,m.ch_u], "features" : m.pv_f},
            "Paneles solares térmicos" : {"index": [m.pt_u], "features" : m.pt_f},
            "Baterías" : {"index" : [m.bat_u,m.ch_u], "features" : m.bat_f},
            "Inversores híbridos"  : {"index" : [m.pv_u,m.bat_u,m.ch_u], "indice_princ" : 2, "features" : m.ch_f},
            "Microturbinas eólicas" : {"index": [m.wt_u], "features" : m.wt_f},
            "Calderas" : {"index" : [m.boi_u], "features" : m.boi_f},
            "Calentadores eléctricos" : {"index" : [m.eh_u], "features" : m.eh_f},
            "CHPs" : {"index" : [m.chp_u], "features" : m.chp_f},
            "Enfriadores de absorsión" : {"index" : [m.ac_u], "features" : m.ac_f},
            "Enfriadores eléctricos" : {"index" : [m.ac_u], "features" : m.ac_f}
        }        
        
        self.inversion_inicial = value(
            sum(m.X_PV[tpv,tch]*(m.pv_f['C_inst',tpv]) for tch in m.ch_u for tpv in m.pv_u)
            + sum(m.X_PT[tpt]*(m.pt_f['C_inst',tpt]) for tpt in m.pt_u)
            + sum(m.X_B[tb,tch]*(m.bat_f['C_inst',tb]) for tch in m.ch_u for tb in m.bat_u)
            + sum(sum(m.X_CH[tpv,tb,tch] for tb in m.bat_u for tpv in m.pv_u)*(m.ch_f['C_inst',tch]) for tch in m.ch_u)
            + sum(m.X_WT[tt]*(m.wt_f['C_inst',tt]) for tt in m.wt_u) 
            + m.d_cost_inst
            + sum(m.X_BOI[tboi]*m.boi_f['C_inst',tboi] for tboi in m.boi_u)
            + sum(m.X_EH[teh]*m.eh_f['C_inst',teh] for teh in m.eh_u)
            + sum(m.X_CHP[tchp]*m.chp_f['C_inst',tchp] for tchp in m.chp_u)
            + sum(m.X_AC[tac]*m.ac_f['C_inst',tac] for tac in m.ac_u)
            + sum(m.X_EC[tec]*m.ec_f['C_inst',tec] for tec in m.ec_u)
        )

        print("Valor inversión inicial: ", self.inversion_inicial)