import pandas as pd
import sklearn
import seaborn as sns
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix

protocolo="n400_frases"

pathusuario1=f"C:/Users/pablo/Desktop/TFG/Experimento/Datos_participantes/Aaron/Características/{protocolo}"
pathusuario2=f"C:/Users/pablo/Desktop/TFG/Experimento/Datos_participantes/Adrian/Características/{protocolo}"
pathusuario3=f"C:/Users/pablo/Desktop/TFG/Experimento/Datos_participantes/Alejandra/Características/{protocolo}"
pathusuario4=f"C:/Users/pablo/Desktop/TFG/Experimento/Datos_participantes/Almudena/Características/{protocolo}"
pathusuario5=f"C:/Users/pablo/Desktop/TFG/Experimento/Datos_participantes/Alvarito/Características/{protocolo}"
pathusuario6=f"C:/Users/pablo/Desktop/TFG/Experimento/Datos_participantes/Andrea/Características/{protocolo}"
pathusuario7=f"C:/Users/pablo/Desktop/TFG/Experimento/Datos_participantes/Guada/Características/{protocolo}"
pathusuario8=f"C:/Users/pablo/Desktop/TFG/Experimento/Datos_participantes/Jandra/Características/{protocolo}"
pathusuario9=f"C:/Users/pablo/Desktop/TFG/Experimento/Datos_participantes/Javier Blanco/Características/{protocolo}"
pathusuario10=f"C:/Users/pablo/Desktop/TFG/Experimento/Datos_participantes/Juan Diego/Características/{protocolo}"
pathusuario11=f"C:/Users/pablo/Desktop/TFG/Experimento/Datos_participantes/Luis/Características/{protocolo}"
pathusuario12=f"C:/Users/pablo/Desktop/TFG/Experimento/Datos_participantes/Miguel/Características/{protocolo}"
pathusuario13=f"C:/Users/pablo/Desktop/TFG/Experimento/Datos_participantes/Mónica/Características/{protocolo}"
pathusuario14=f"C:/Users/pablo/Desktop/TFG/Experimento/Datos_participantes/Nieves/Características/{protocolo}"
pathusuario15=f"C:/Users/pablo/Desktop/TFG/Experimento/Datos_participantes/Perales/Características/{protocolo}"
pathusuario16=f"C:/Users/pablo/Desktop/TFG/Experimento/Datos_participantes/Sara/Características/{protocolo}"
pathusuario17=f"C:/Users/pablo/Desktop/TFG/Experimento/Datos_participantes/Sergio/Características/{protocolo}"
pathusuario18=f"C:/Users/pablo/Desktop/TFG/Experimento/Datos_participantes/Sergio Aranda/Características/{protocolo}"
pathusuario19=f"C:/Users/pablo/Desktop/TFG/Experimento/Datos_participantes/Vicente/Características/{protocolo}"
pathusuario20=f"C:/Users/pablo/Desktop/TFG/Experimento/Datos_participantes/Yuri/Características/{protocolo}"

# Lista de rutas de archivos
array_paths = [pathusuario1, pathusuario2, pathusuario3, pathusuario4, pathusuario5, pathusuario6, pathusuario7, pathusuario8, pathusuario9 , pathusuario10,
               pathusuario11,  pathusuario12, pathusuario13, pathusuario14, pathusuario15, pathusuario16, pathusuario17, pathusuario18, pathusuario19, pathusuario20]

for u in range (len(array_paths)):
    # Valores para la columna 'label'
    num_usuarios = len(array_paths)
    labels = [0] * num_usuarios
    usuario_a_clasificar = u # Índice del usuario a autenticar; CAMBIAR según corresponda
    print("USUARIO"+str(u+1) +"\n")
    labels[usuario_a_clasificar] = 1

    # Lista para almacenar los DataFrames
    dataframes = []
    num_filas = []

    # Leer los archivos y agregar la columna 'label'
    for i, path in enumerate(array_paths):
        df = pd.read_csv(path).T
        df['label'] = labels[i]
        dataframes.append(df)
        num_filas.append(df.shape[0])

    # Determinar el número mínimo de columnas
    min_rows = min(num_filas)

    # Truncar todas las tablas al número mínimo de columnas, incluyendo 'label'
    for i in range(len(dataframes)):
        dataframes[i] = dataframes[i].iloc[:min_rows, :].copy()  # Truncar las columnas de datos
        dataframes[i]['label'] = labels[i]  # Asegurarse de incluir la columna 'label'

    # Concatenar todos los DataFrames en uno solo
    dataset = pd.concat(dataframes, axis=0).reset_index(drop=True) 

    # Rellenar valores faltantes con cero
    dataset = dataset.fillna(0)
    #Solo para SVM (quedarse con solo el canal TP9 y AF7)
    #indices1=np.arange(1,dataset.shape[0],4)
    #indices2=np.arange(2,dataset.shape[0],4)
    #indices=np.concatenate((indices1, indices2))
    #indices= np.sort(indices)
    #dataset = dataset.iloc[indices]

    #print(dataset)

    # Dividir en características (X) y etiquetas (y) y dividir el dataset en cojunto de entrenammiento y prueba
    x = dataset.drop('label',axis=1) 
    y = dataset['label']
    x_train,x_test, y_train,y_test=sklearn.model_selection.train_test_split(x,y,test_size=0.3)

    

    # SVM
    model_svm = SVC(kernel='linear') # Inicializar el modelo SVM con un kernel lineal
    model_svm.fit(x_train, y_train) # Entrenar el modelo SVM
    y_pred_svm = model_svm.predict(x_test)# Hacer predicciones sobre los datos de prueba
    accuracy_svm = metrics.accuracy_score(y_test, y_pred_svm) # Evaluar la precisión del modelo SVM
    print("Accuracy SVM:", accuracy_svm)
    cnf_matrix_SVM = metrics.confusion_matrix(y_test, y_pred_svm)
    TN = cnf_matrix_SVM[0][0]
    TP = cnf_matrix_SVM[1][1]
    FP = cnf_matrix_SVM[0][1]
    FN = cnf_matrix_SVM[1][0]
    print("Tasa FP_SVM",(FP / ( FP + TN)))
    print("Tasa FN_SVM",(FN / (FN + TP)))
    print("Tasa acierto_SVM",TP/(TP+FN))



    """
    # LDA
    model_lda = LinearDiscriminantAnalysis() # Inicializar el modelo LDA
    model_lda.fit(x_train, y_train) #Entrenar el modelo
    y_pred_lda = model_lda.predict(x_test)
    accuracy_lda = metrics.accuracy_score(y_test, y_pred_lda)
    print("Accuracy LDA:", accuracy_lda)

    cnf_matrix_LDA = metrics.confusion_matrix(y_test, y_pred_lda)
    TN = cnf_matrix_LDA[0][0]
    TP = cnf_matrix_LDA[1][1]
    FP = cnf_matrix_LDA[0][1]
    FN = cnf_matrix_LDA[1][0]
    print(TN,TP,FP,FN)
    print("Tasa FP_LR",(FP / ( FP + TN))) # Tasa falso positivo
    print("Tasa FN_LR",(FN / (FN + TP))) #Tasa falso negativo
    print("Tasa acierto_LR",TP/(TP+FN),"\n") #Tasa positivos

    # KNN
    model_knn = KNeighborsClassifier(n_neighbors=5) # Inicializar el modelo KNN con k=5 (puedes ajustar el valor de k según tus necesidades)
    model_knn.fit(x_train, y_train) # Entrenar el modelo KNN
    y_pred_knn = model_knn.predict(x_test) # Hacer predicciones sobre los datos de prueba
    accuracy_knn = metrics.accuracy_score(y_test, y_pred_knn) # Evaluar la precisión del modelo KNN
    print("Accuracy KNN:", accuracy_knn)

    cnf_matrix_KNN = metrics.confusion_matrix(y_test, y_pred_knn)
    TN = cnf_matrix_KNN[0][0]
    TP = cnf_matrix_KNN[1][1]
    FP = cnf_matrix_KNN[0][1]
    FN = cnf_matrix_KNN[1][0]
    print(TN,TP,FP,FN)
    print("Tasa FP_KNN",(FP / ( FP + TN)))
    print("Tasa FN_KNN",(FN / (FN + TP)))
    print("Tasa acierto_KNN",TP/(TP+FN),"\n")
    """    
        


    '''''
    # TNN
    # Normalizar los datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(x_train)
    X_test_scaled = scaler.transform(x_test)
    # Definir la arquitectura de la red neuronal
    input_shape = X_train_scaled.shape[1] #Se determina la forma de las entradas de la red neuronal gemela, que corresponde al número de filas en los datos de entrada.
    output_shape = len(np.unique(y_train)) 
    # Definir la capa de entrada
    input_layer = tf.keras.layers.Input(shape=(input_shape,)) # Se define la capa de entrada de la red neuronal gemela, que especifica el tamaño de las entradas.
    # Definir las capas gemelas
    hidden_layer1 = tf.keras.layers.Dense(64, activation='relu')(input_layer) # Esta capa se conecta a la capa de entrada y tiene 64 neuronas con una función de activación ReLU
    hidden_layer2 = tf.keras.layers.Dense(64, activation='relu')(input_layer) # Idéntica a la primera capa oculta pero independiente
    # Combinar las salidas de las capas gemelas
    merged_layer = tf.keras.layers.Concatenate()([hidden_layer1, hidden_layer2])
    # Crear la capa de salida
    output_layer = tf.keras.layers.Dense(output_shape, activation='softmax')(merged_layer)
    # Definir el modelo
    model_tnn = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
    # Compilar el modelo
    model_tnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # Entrenar el modelo
    model_tnn.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_split=0.2) #????
    y_pred_tnn = model_tnn.predict(X_test_scaled)
    # Evaluar el modelo
    test_loss, test_accuracy = model_tnn.evaluate(X_test_scaled, y_test)
    print('TNN Test accuracy:', test_accuracy)

    cnf_matrix_SVM = metrics.confusion_matrix(y_test, y_pred_tnn)
    TN = cnf_matrix_SVM[0][0]
    TP = cnf_matrix_SVM[1][1]
    FP = cnf_matrix_SVM[0][1]
    FN = cnf_matrix_SVM[1][0]
    print("Tasa FP_SVM",(FP / ( FP + TN)))
    print("Tasa FN_SVM",(FN / (FN + TP)))
    print("Tasa acierto_SVM",TP/(TP+FN))

    '''''
