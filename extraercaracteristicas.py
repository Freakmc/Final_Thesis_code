import numpy as np
from tabulate import tabulate
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg

def extracción_carácterísticas(ruta_época_a_extraer_las_características):

    df = pd.read_csv(ruta_época_a_extraer_las_características)
    df =  df.fillna(df.mean())

    # Auto Regressive (time domain)
    coef_AR=[] # Lista para almacenar los coeficientes AR de la época
    for column in range(1,5): #para cada columna
        canal=df.iloc[:,column]
        pd.plotting.autocorrelation_plot(canal)
        train_data = canal.head(round(len(canal)*0.7))
        test_data = canal.tail(round(len(canal)*0.3))
        model = AutoReg(train_data, lags=23)
        model_fitted = model.fit()
        coef_AR.append([])
        for j in range (len(model_fitted.params)-1):
            coef_AR[column-1].append(model_fitted.params[j])

    # Psd spectrum density (frequency domain)
    coef_psd=[]
    for column in range(1,5):
        canal=df.iloc[:,column]
        coef_psd.append([])
        psd_tramos=[]
        eeg_bands = {'Alpha': (8, 12),'Beta': (12, 30),'Gamma': (30, 50)} # Definir las bandas EEG 

        #Para cada quinto de 200 ms
        for i in range(5):
            inicio_tramo=i*51 #(256hz/5)
            tramo=np.array(canal[inicio_tramo:inicio_tramo+51]) #Array con cada quinto
            fft=np.absolute(np.fft.rfft(tramo))**2 #fast furier transform de cada tramo
            fft_freq = np.fft.rfftfreq(51, 1/256) # Get frequencies for amplitudes in Hz   

            for band in eeg_bands:  
                freq_index = np.where((fft_freq >= eeg_bands[band][0]) & (fft_freq <= eeg_bands[band][1]))[0]
                eeg_band_fft = np.mean(fft[freq_index]) 
                psd_tramos.append(eeg_band_fft)

        #Para toda la epoca
        fft = np.absolute(np.fft.rfft(canal))**2
        fft_freq = np.fft.rfftfreq(len(canal), 1/256) # Get frequencies for amplitudes in Hz    

        for band in eeg_bands:  
            freq_ix = np.where((fft_freq >= eeg_bands[band][0]) & (fft_freq <= eeg_bands[band][1]))[0]
            eeg_band_fft = np.mean(fft[freq_ix]) # Take the mean of the fft amplitude for each EEG band
            psd_tramos.append(eeg_band_fft)

        # Unir todos los coeficientes
        for j in range(len(psd_tramos)):
            coef_psd[column-1].append(psd_tramos[j])

    # Crear la matriz con todas las características de la época
    coeficientes_epoca=[] 
    for column in range(1,5):
        coeficientes_epoca.append(coef_AR[column-1]+coef_psd[column-1])

    return pd.DataFrame(coeficientes_epoca).T