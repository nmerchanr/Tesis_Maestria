import ssl
import pandas as pd
import numpy as np
from numpy import radians as r
import altair as alt
import plotly.graph_objects as go
import plotly.subplots as sp
import ipywidgets as widgets
from IPython.display import display

alt.data_transformers.disable_max_rows()

def tabbed_df_viewer(df_dict : dict):
    """
    Muestra los DataFrames en un widget de pestañas.
    
    :param df_dict: Diccionario donde las llaves son los nombres de los DataFrames y los valores son los DataFrames.
    """
    # Crear un Tab widget
    tab = widgets.Tab()

    # Crear una lista de widgets de salida, uno para cada DataFrame
    children = []    
    for name, df in df_dict.items():          
        output = widgets.Output()           
        output.append_display_data(df)   
            
        children.append(output)
        

    # Asignar las salidas a las pestañas
    tab.children = children

    # Nombrar cada pestaña según la llave del diccionario
    for i, name in enumerate(df_dict.keys()):
        tab.set_title(i, name)

    # Mostrar las pestañas
    display(tab)


def tabbed_df_viewer(df_dict):
    """
    Muestra los DataFrames en un widget de pestañas.
    
    :param df_dict: Diccionario donde las llaves son los nombres de los DataFrames y los valores son los DataFrames.
    """
    # Crear un Tab widget
    tab = widgets.Tab()

    # Crear una lista de widgets de salida, uno para cada DataFrame
    children = []
    for name, df in df_dict.items():
        output = widgets.Output()
        with output:
            display(df)
        children.append(output)

    # Asignar las salidas a las pestañas
    tab.children = children

    # Nombrar cada pestaña según la llave del diccionario
    for i, name in enumerate(df_dict.keys()):
        tab.set_title(i, name)

    # Mostrar las pestañas
    display(tab)

def get_data_fromNSRDB(lat, lon, year):
    
    ssl._create_default_https_context = ssl._create_unverified_context
    # You must request an NSRDB api key from the link above
    api_key = 'cTc2xIqsUEZws0YRXLH2wgfu4HL6ifazGnQJFp50'
    # Set the attributes to extract (e.g., dhi, ghi, etc.), separated by commas.
    attributes = 'ghi,dhi,dni,wind_speed,air_temperature,solar_zenith_angle'
    # Choose year of data
    #year = '2019'
    # Set leap year to true or false. True will return leap day data if present, false will not.
    leap_year = 'false'
    # Set time interval in minutes, i.e., '30' is half hour intervals. Valid intervals are 30 & 60.
    interval = '30'
    # Specify Coordinated Universal Time (UTC), 'true' will use UTC, 'false' will use the local time zone of the data.
    # NOTE: In order to use the NSRDB data in SAM, you must specify UTC as 'false'. SAM requires the data to be in the
    # local time zone.
    utc = 'false'
    # Your full name, use '+' instead of spaces.
    your_name = 'Nicolas+Merchan'
    # Your reason for using the NSRDB.
    reason_for_use = 'beta+testing'
    # Your affiliation
    your_affiliation = 'universidad+nacional'
    # Your email address
    your_email = 'nmerchanr@unal.edu.co'
    # Please join our mailing list so we can keep you up-to-date on new developments.
    mailing_list = 'false'

    # Declare url string
    url = 'https://developer.nrel.gov/api/solar/nsrdb_psm3_download.csv?wkt=POINT({lon}%20{lat})&names={year}&leap_day={leap}&interval={interval}&utc={utc}&full_name={name}&email={email}&affiliation={affiliation}&mailing_list={mailing_list}&reason={reason}&api_key={api}&attributes={attr}'.format(year=year, lat=lat, lon=lon, leap=leap_year, interval=interval, utc=utc, name=your_name, email=your_email, mailing_list=mailing_list, affiliation=your_affiliation, reason=reason_for_use, api=api_key, attr=attributes)

    df = pd.read_csv(url, header=2)
    df = df.loc[df['Minute'] == 0, :]
    info = pd.read_csv(url, nrows=1)
    
    return df, info

def calculate_WT_power(df, WindGens, z0, height, elevation):
    Turbine_profile = pd.DataFrame(columns = WindGens.columns, index = range(1, len(df)+1))
    Profiles = {}
    
    coef_den = ((1-(0.0065*elevation)/288.16)**(9.81/(287*0.0065)))*(288.16/(288.16-0.0065*elevation))    
    
    for i in Turbine_profile.columns:
        Profiles[i] = pd.read_excel('Catalogo.xlsx',sheet_name = i,header=0,index_col=0)
        #height = WindGens.loc["h", i]
        min_vel = WindGens.loc["v_st",i]
        max_vel = Profiles[i].index[-1]
        break_vel = WindGens.loc["v_max",i]

        for k, ws_value_st in enumerate(df["Wind Speed"].values, 1):

            #ws_value = ws_value_st*((height/10)**alpha_wind)
            

            ws_value = ws_value_st*(np.log(height/z0)/np.log(10/z0))
            if (ws_value <= min_vel) or (ws_value > break_vel):
                Turbine_profile.loc[k, i] = 0
            elif ws_value > max_vel:
                Turbine_profile.loc[k, i] = Profiles[i]["power"].iloc[-1]*coef_den
            else:
                new_index = np.sort(np.append(Profiles[i].index.values, ws_value))        
                Turbine_profile.loc[k, i] = Profiles[i].reindex(new_index).interpolate(method='linear').loc[ws_value,"power"]*coef_den

    return Profiles, Turbine_profile

def power_PV_calculation(df_meteo, PVtype, azimut, inc_panel, lat):
    
    df = df_meteo.copy()

    azimut, inc_panel = r(azimut), r(inc_panel)

    df["Day of year"] = df.index.day_of_year
    df['decl'] = 23.45*np.sin(2*np.pi*(284+df['Day of year'].to_numpy())/365)
    df['omega'] = 15*(df['Hour'].to_numpy() - 12)

    decl = r(df['decl'].to_numpy())
    omega = r(df['omega'].to_numpy())
    lat_r = r(lat)

    df['tetha'] = np.arccos(np.sin(decl)*np.sin(lat_r)*(np.cos(inc_panel)-np.cos(azimut)) +
                            np.cos(decl)*np.cos(lat_r)*np.cos(inc_panel)*np.cos(omega) +
                            np.cos(decl)*np.sin(lat_r)*np.sin(inc_panel)*np.cos(azimut)*np.cos(omega) +
                            np.cos(decl)*np.sin(inc_panel)*np.sin(azimut)*np.sin(omega))

    df['zenit'] = np.arccos(np.cos(lat_r)*np.cos(decl)*np.cos(omega) +
                        np.sin(lat_r)*np.sin(decl))

    Rb = np.cos(df['tetha'].to_numpy())/np.cos(df['zenit'].to_numpy())

    df['IRR'] = df['GHI'].to_numpy()*Rb
    
    df['IRR'] = df['IRR'].where(df['IRR']>0, 0)
    
    Tm = df['IRR'].values*np.exp(-3.47-0.0594*df['Wind Speed'].values)+df['Temperature'].values

    T_panel = Tm+(df['IRR'].values/1000)*3

    P_mpp = pd.DataFrame(index = df.index, columns = PVtype.columns)

    for k in list(PVtype.columns):
        P_mpp[k] = (PVtype.loc['P_stc',k]*(1+(PVtype.loc['Tc_Pmax',k]/100)*(T_panel-25))*(df['IRR'].values/1000))/1000
                
    return P_mpp

import json, requests

def extract_tem_min(lat,lon):

    base_url = r"https://power.larc.nasa.gov/api/temporal/climatology/point?parameters=T2M,T2M_MAX,T2M_MIN&community=SB&longitude={longitude}&latitude={latitude}&format=JSON"
    api_request_url = base_url.format(longitude=lon, latitude=lat)
    response = requests.get(url=api_request_url, verify=True, timeout=30.00)
    temp_hist = json.loads(response.content.decode('utf-8'))
        
    Temp_min = temp_hist['properties']['parameter']['T2M_MIN']['ANN']

    return Temp_min


def createfig_heatmap(df, col, binary, units, title):
    fechas = pd.date_range(start = '01/01/2019 00:00', end='31/12/2019 23:00', freq='1H')
    df_fig = df.copy()
    df_fig['Hora'] = fechas.hour
    df_fig['day_of_year'] = fechas.dayofyear
    df_fig['tooltip'] = "D: " + df_fig["day_of_year"].map(str) + ", H: " + df_fig["Hora"].map(str) + ", Val: " + df_fig[col].map(str)

    if binary:
        color = alt.Color(col + ':Q', title=units, scale=alt.Scale(domain=[0, 1],range=['#030303', '#239B56'],type='linear'))
    else:
        color = alt.Color(col + ':Q', title=units)

    fig = alt.Chart(df_fig, title =title).mark_rect().encode(x=alt.X('day_of_year:O', title='Día del año', axis=alt.Axis(values=np.arange(0,366,25), labelAngle=0)), y=alt.Y('Hora:O', title='Hora'), color=color, tooltip='tooltip:N').configure_axis(
                                                labelFontSize=22,
                                                titleFontSize=22
                                            ).properties(
                                            width=1300,
                                            height=500
                                        ).configure_scale(
                                            bandPaddingInner=0.03
                                        ).configure_title(fontSize=24)
    
    return fig


def plot_time_series(df, time_col):
    # Verifica que la columna temporal esté en el DataFrame
    if time_col not in df.columns:
        raise ValueError(f"La columna '{time_col}' no está en el DataFrame.")

    # Elimina la columna temporal de las columnas a graficar
    columns_to_plot = [col for col in df.columns if col != time_col]

    # Crear una figura con subplots
    num_cols = len(columns_to_plot)
    rows = (num_cols + 2) // 3  # Ajusta el número de filas
    fig = sp.make_subplots(rows=rows, cols=3, subplot_titles=columns_to_plot)

    for i, col in enumerate(columns_to_plot):
        row = i // 3 + 1
        col_idx = i % 3 + 1
        
        # Verifica si los valores son binarios
        if df[col].dropna().isin([0, 1]).all():
            # Gráfico de barras
            fig.add_trace(
                go.Bar(x=df[time_col], y=df[col], name=col),
                row=row, col=col_idx
            )
        else:
            # Gráfico de líneas
            fig.add_trace(
                go.Scatter(x=df[time_col], y=df[col], mode='lines', name=col),
                row=row, col=col_idx
            )
    
    fig.update_layout(height=400 * rows, title_text="Series de Tiempo")
    fig.show()
