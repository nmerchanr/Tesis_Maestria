import os
import json
import re
import pandas as pd
import numpy as np
import pickle

from datetime import datetime

import mpisppy.utils.sputils as sputils
from mpisppy.opt.lshaped import LShapedMethod
from mpisppy.opt.ef import ExtensiveForm

from utils.functions import power_PV_calculation, calculate_WT_power, get_data_fromNSRDB
from utils.model import create_model

from pyomo.environ import value
import pyomo.environ as pyo
import unicodedata



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

        self.ls.first_stage_solution_available = True
        self.ls.tree_solution_available = True

        os.makedirs(self.ruta_resultados, exist_ok=True)        
        self.ls.write_tree_solution(self.ruta_resultados, self.scenario_tree_solution_writer)

        with open(f"{self.ruta_resultados}/data_model.pkl", "wb") as f:
            pickle.dump(self.data_model, f)

        print(f"Resultados guardados con exito en: {self.ruta_resultados}")
    

    def LS_optimizacion(self, max_iter : int = 65, bound = 0, method = "LS", mipgap = 0.01): 
        
        
        if method == "LS":
            options = options = {
                "root_solver": "cplex", 
                "sp_solver": "cplex", 
                "tol": 1e-3, 
                "sp_solver_options": { 
                    "parallel": -1, 
                    "mip tolerances mipgap": mipgap                    
                },
                "max_iter": max_iter, 
                "valid_eta_lb": {name: bound for name in self.escenarios}, 
                "verbose": True 
            }

            self.ls = LShapedMethod(options, self.escenarios, self.scenario_creator)
            result = self.ls.lshaped_algorithm()

        elif method == "EF":
            options = {
                "solver": "cplex"
            }

            solver_options = {
                'mip tolerances mipgap' : mipgap
            }

            self.ls = ExtensiveForm(options, self.escenarios, self.scenario_creator)
            results = self.ls.solve_extensive_form(solver_options=solver_options, tee = True) 
        
        self.guardar_resultados()            


class Maestro_Resultados(Maestro_Optimizacion):

    def __init__(self, location : dict,  estocastico : bool):       
        
        if estocastico:
            self.ruta_resultados = f'RESULTADOS/{location["name_data"]}/estocastico'
        else:
            self.ruta_resultados = f'RESULTADOS/{location["name_data"]}/determinista'

        with open(f"{self.ruta_resultados}/data_model.pkl", "rb") as f:
            data_model = pickle.load(f)

        data_model["needs"]["n"] = self.quitar_tildes_df(data_model["needs"]["n"], indice=True) 
        data_model["needs"]["load"] = self.quitar_tildes_df(data_model["needs"]["load"], indice=True)

        df_meteo, info_meteo = get_data_fromNSRDB(location["lat"], location["lon"], location["year_deterministic"])
        date_vec = np.vectorize(datetime)
        df_index = date_vec(df_meteo.Year.values,df_meteo.Month.values,df_meteo.Day.values, df_meteo.Hour.values, df_meteo.Minute.values, tzinfo=None)
        df_meteo.index = df_index

        super().__init__(data_model, location, df_meteo, info_meteo, estocastico)

        self.leer_resultados()  


    def quitar_tildes_df(self, df : pd.DataFrame, columnas=None, indice=False):
        """
        Elimina las tildes de los valores string en un DataFrame.

        Parámetros:
        - df (pd.DataFrame): DataFrame original.
        - columnas (list, opcional): Lista de columnas donde eliminar tildes. Si es None, se aplicará a todas.
        - indice (bool, opcional): Si es True, también eliminará tildes del índice.

        Retorna:
        - pd.DataFrame: DataFrame con los valores sin tildes.
        """        

        df_modificado = df.copy()  # Para no modificar el original
        
        # Aplicar a columnas específicas o a todas si no se especifican
        if columnas is None:
            columnas = df.columns  # Usar todas las columnas si no se especifica
        
        df_modificado.columns = df_modificado.columns.map(self.quitar_tildes)

        # Aplicar al índice si es necesario
        if indice:
            df_modificado.index = df_modificado.index.map(self.quitar_tildes)

        return df_modificado
    
    def quitar_tildes(self, s):
        
        if isinstance(s, str):  # Solo procesar strings
            return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
        return s  # Devolver otros tipos de datos sin cambios
    
    def corregir_indices(self, cadena : str):
        # Expresión regular para encontrar los índices dentro de los corchetes
        patron = r"\[([^\]]+)\]"  # Captura lo que está dentro de los corchetes

        cadena = self.quitar_tildes(cadena)

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
                        if abs(item["valor"]) < 0.001:
                            valor = 0
                        else:
                            valor = item["valor"]                

                    exec(f'model.{variable}.value = {valor}')

                else: 
                    print("Variable no encontrada", variable)                      

            self.model_escenarios[esc] = model
    