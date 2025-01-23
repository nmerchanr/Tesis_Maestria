import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime
import locale
import os
locale.setlocale(locale.LC_ALL, 'es_ES')

def create_weather_BMG_scens(df, pdf_scipy, col, seed = None):

    if seed is not None:
        np.random.seed(seed=seed)
    
    fechas = pd.date_range(start = '01/01/2019 00:00', end='31/12/2019 23:00', freq='1H')
    
    df_scens = pd.DataFrame(columns=["bad","mean","good"])
    df_scens["hour"] = fechas.hour
    df_scens["month"] = fechas.month
    
    scens_prob = {"bad":[],"mean":[],"good":[]}
    
    for month in range(0,12):        
        for hour in range(0,24):
                                    
            data = df.loc[np.logical_and(df.Hour == hour, df.Month == month + 1) , col].to_numpy()                        
            
            num_hours = np.sum(np.logical_and(fechas.month == month + 1, fechas.hour == hour)) 
            max_value = np.max(data)
            
            if (data == 0).all():
                for key in ["bad","mean","good"]:
                    df_scens.loc[np.logical_and(df_scens["hour"] == hour, df_scens["month"] == month + 1), key] = 0               
            else: 
                params = pdf_scipy.fit(data, loc = 0)
                dist = pdf_scipy(*params)
                
                mean, std = dist.mean(), dist.std()
                
                scens_prob["bad"].append(dist.cdf(mean - std))
                scens_prob["mean"].append(dist.cdf(mean + std) - scens_prob["bad"][-1])
                scens_prob["good"].append(1 - scens_prob["bad"][-1] - scens_prob["mean"][-1])
                
                scen_vals = {}
                cont = 0
                while True:
                    vals = np.round(dist.rvs(size = 10000),2)

                    scen_vals["bad"] = vals[np.logical_and(vals <= mean - std, vals > 0)]
                    scen_vals["mean"] = vals[np.logical_and(vals > mean - std, vals < mean + std)]
                    scen_vals["good"] = vals[np.logical_and(vals >= mean + std, vals < max_value)]

                    if len(scen_vals["bad"]) > num_hours and len(scen_vals["mean"]) > num_hours and len(scen_vals["good"]) > num_hours:
                        for key in ["bad","mean","good"]:
                            df_scens.loc[np.logical_and(df_scens["hour"] == hour, df_scens["month"] == month + 1), key] = scen_vals[key][0:num_hours]   
                        break
                    if cont > 5000:
                        for key in ["bad","mean","good"]:
                            df_scens.loc[np.logical_and(df_scens["hour"] == hour, df_scens["month"] == month + 1), key] = data[0:num_hours]
                        print("Could not assign random data. Historical assignees")
                        break

                    cont += 1

                
    for key in ["bad","mean","good"]:
        scens_prob[key] = np.mean(scens_prob[key])
    
    return df_scens, scens_prob

def plot_pdf(
        df: pd.DataFrame, 
        pdf, 
        month: int = 1,
        hour: int = 1,
        col_data: str = "", 
        name_var: str = "",
        units: str = "", 
        location=None
    ):
    """
    Genera una figura combinada con:
    - Dos subplots en la izquierda (verticales) para los gráficos de la columna seleccionada "col_data".
    - Una cuadrícula de subplots en la derecha para PDFs por hora.
    """
    matplotlib.rcParams.update({'font.size': 16})   

    font_size = 22

    month_name = datetime(1900, month, 1).strftime('%B')
    years = [2015, 2016, 2017, 2018, 2019, 2020]

    # Crear figura principal
    fig = plt.figure(figsize=(40, 25))  # Tamaño ajustado para integrar todos los gráficos

    # Subplot Izquierdo Superior: de la columna seleccionada "col_data" por mes
    ax1 = plt.subplot2grid((4,9), (0, 0), rowspan=2, colspan=3)
    for year in years:
        mask = np.logical_and(df["Year"] == year, df["Month"] == month)
        x = df.index[mask]
        y = df.loc[mask, col_data].to_numpy()
        ax1.plot(x, y, label=f'{year}')
    ax1.set_title(f'{name_var} en {location["name"].capitalize()}. {month_name}',fontsize=font_size, fontweight='bold')
    ax1.set_ylabel(f'${units}$',fontsize=font_size)
    ax1.legend(fontsize=font_size)
    ax1.tick_params(axis='x', labelsize=font_size) 
    ax1.tick_params(axis='y', labelsize=font_size) 
    ax1.grid()

    # Subplot Izquierdo Inferior: de la columna seleccionada "col_data" a una hora específica
    ax2 = plt.subplot2grid((4,9), (2, 0), rowspan=2, colspan=3)
    for year in years:
        mask = np.logical_and.reduce([df["Year"] == year, df["Month"] == month, df["Hour"] == hour])
        x = df.index[mask]
        y = df.loc[mask, col_data].to_numpy()
        ax2.scatter(x, y, label=f'{year}', alpha=0.7)
    ax2.set_title(f'{name_var} en {location["name"].capitalize()}. {month_name} a las {hour}:00',fontsize=font_size, fontweight='bold')
    ax2.set_xlabel("Fecha",fontsize=font_size)
    ax2.set_ylabel(f'${units}$',fontsize=font_size)
    ax2.legend(fontsize=font_size)
    ax2.tick_params(axis='x', labelsize=font_size) 
    ax2.tick_params(axis='y', labelsize=font_size) 
    ax2.grid()

    for spine in ax2.spines.values():  # ax2 es el eje al que deseas resaltar
        spine.set_edgecolor('red')  # Color del recuadro
        spine.set_linewidth(4)  # Grosor del recuadro

    # Subplot Derecho: Figura de PDFs
    hour_tmp = 0
    for row in range(0, 4):
        for col in range(0, 6):  # Usar 3 columnas para los subplots de la derecha
            ax_pdf = plt.subplot2grid((4,9), (row, col + 3))  # Inicia en la cuarta columna
            data = df.loc[np.logical_and(df.Hour == hour_tmp, df.Month == month), col_data].to_numpy()

            # Mostrar el histograma y la PDF
            ax_pdf.hist(data, bins=40, density=True)

            params = pdf.fit(data, loc=0)
            dist = pdf(*params)

            x = np.linspace(dist.ppf(0.01), dist.ppf(0.9999), 10000)
            y1 = dist.pdf(x)

            mean, std = dist.mean(), dist.std()

            ax_pdf.plot(x, y1, linewidth=2.5, label="pdf")
            ax_pdf.axvline(x=mean, linewidth=2.5, color='#F75D3C', label='media')
            ax_pdf.axvline(x=mean + std, linewidth=2.5, color='#58F73C', label='media + std')
            ax_pdf.axvline(x=mean - std, linewidth=2.5, color='#F73CF7', label='media - std')

            if hour_tmp == hour:
                for spine in ax_pdf.spines.values():  
                    spine.set_edgecolor('red') 
                    spine.set_linewidth(4)  


            ax_pdf.set_title(f'Hora {hour_tmp}:00',fontsize=font_size, fontweight='bold')
            ax_pdf.tick_params(axis='x', labelsize=font_size) 
            ax_pdf.tick_params(axis='y', labelsize=font_size) 
            ax_pdf.grid()
            ax_pdf.legend(fontsize=16)
            ax_pdf.set_xlabel(f'${units}$', loc='right',fontsize=font_size, labelpad=0)

             
            
            hour_tmp += 1

    path = "Figuras/pdf"
    os.makedirs(path, exist_ok=True) 

    plt.savefig(f'{path}/pdf_{location["name"]}_{col_data}.pdf', format='pdf', transparent=True)

    return fig
    
def create_av_scens(data, seed = None):

    if seed is not None:
        np.random.seed(seed=seed)

    SAIFI = np.sum(data != 24)
    SAIDI = np.mean(24-data[data != 24])
    print(f'SAIDI: {SAIDI}, SAIFI {SAIFI}')

    df_scens = pd.DataFrame(columns=["scen1","scen2"], index = range(0,8760))    
    
    scens_prob = {"scen1":0.5,"scen2":0.5}

    for key in ["scen1","scen2"]:
        prob_saifi = np.random.normal(SAIFI, np.sqrt(SAIFI), 1)

        dias = []

        for i in range(0, round(prob_saifi[0])):
            dias.append(np.random.randint(0,8760-1))


        prob_saidi = np.abs(np.random.normal(SAIDI, np.sqrt(SAIDI), len(dias)))
        
        disp = np.ones(8760)

        for k, d in enumerate(dias):
            disp[d:d + int(np.round(prob_saidi[k]))] = 0

        df_scens[key] = disp

    return df_scens, scens_prob