import pylsl
import csv
import sys

def capturar_datos_eeg(filename):
    try:
        # Crear un archivo CSV para guardar los datos
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)               
            # Obtener el stream EEG de Muse 2
            streams = pylsl.resolve_byprop('type', 'EEG') # Utiliza la biblioteca pylsl para buscar y resolver un flujo de datos EEG por sus propiedades. pylsl es una biblioteca de Python para trabajar con el Protocolo de Enlace de Laboratorio (LSL, por sus siglas en inglés), que es un protocolo para transmitir datos de tiempo real entre dispositivos y software en tiempo real
            if not streams:
                raise RuntimeError ("No se encontró ningún stream EEG.")
            inlet = pylsl.StreamInlet(streams[0])  
            # Obtener la información del stream (número de canales, frecuencia de muestreo, etc.)
            info = inlet.info()
            num_channels = info.channel_count()
            sample_rate = info.nominal_srate()                           
            # Escribir las etiquetas de los canales en la primera fila del archivo CSV 
            channel_labels = ['Channel ' + str(i) for i in range(1, num_channels+1)]
            writer.writerow(['Timestamp'] + channel_labels)   
            # Capturar los datos durante la duración especificada
            #start_time = pylsl.local_clock()
            #end_time = start_time + duration
            #while pylsl.local_clock() < end_time:
            while True:
                # Obtener los datos del EEG
                sample, timestamp = inlet.pull_sample()
                # Escribir los datos en el archivo CSV
                writer.writerow([timestamp] + sample)
    except Exception as e:
        print("Error:",e) 

capturar_datos_eeg(sys.argv[1])   