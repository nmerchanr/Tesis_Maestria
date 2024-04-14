import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

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

def plot_pdf(df,pdf,col_data,month = 1,units=""):
    matplotlib.rcParams.update({'font.size': 15})

    fig, ax = plt.subplots(4,6, figsize=(40,28))

    hour = 0
    for row in range(0,4):
        for col in range(0,6):
            data = df.loc[np.logical_and(df.Hour == hour, df.Month == month) ,col_data].to_numpy()
            
            ax[row,col].hist(data, bins=40,density=True)
            
            params = pdf.fit(data, loc = 0)
            dist = pdf(*params)  
            

            x = np.linspace(dist.ppf(0.01),
                    dist.ppf(0.9999),10000)
            y1 = dist.pdf(x)

            mean, std = dist.mean(), dist.std()
            
            ax[row,col].plot(x,y1, linewidth=3.5, label = "pdf")
            ax[row,col].axvline(x = mean, linewidth=3.5, color = '#F75D3C', label = 'mean')
            ax[row,col].axvline(x = mean + std, linewidth=3.5, color = '#58F73C', label = 'mean + std')
            ax[row,col].axvline(x = mean - std, linewidth=3.5, color = '#F73CF7', label = 'mean - std')
            
            ax[row,col].set_title(f'Hora {hour}:00')
            ax[row,col].grid()
            ax[row,col].legend()
            
            x1 = np.linspace(mean + std, x[-1] , 100)
            #ax[row,col].fill_between(x1, 0, dist.pdf(x1), color = 'r')
            ax[row,col].set_xlabel(units)
            hour += 1

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