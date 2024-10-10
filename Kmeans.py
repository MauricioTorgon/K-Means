from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
import random
from openpyxl import Workbook
from tkinter import *
import tkinter as tk

# fetch dataset 
breast_cancer = fetch_ucirepo(id=14) 

# data (as pandas dataframes) 
X = breast_cancer.data.features

# Definir si los atributos son categóricos o continuos
tipos_datos = [
    #True,  # class
    True,  # age
    True,  # menopause
    True,  # tumor_size (en realidad son rangos categóricos)
    True,  # inv_nodes (también son rangos categóricos)
    True,  # node_caps
    False, # deg_malig (continua)
    True,  # breast
    True,  # breast_quad
    True   # irradiat
]

# Calcular los rangos de los atributos continuos
def calcular_rango(X, tipos_datos):
    rangos = {}
    for col, es_categorico in zip(X.columns, tipos_datos): #zip devuelve tuplas con (nombre_columna,tipos_datos)
        if not es_categorico:  # Si es continuo
            columna_numerica = pd.to_numeric(X[col], errors='coerce') #manejo de string a numerico
            rango = columna_numerica.max() - columna_numerica.min()
            rangos[col] = rango
    return rangos

# Obtener los rangos para las columnas continuas
rangos_continuos = calcular_rango(X, tipos_datos)

# Función HEOM ajustada para incluir el rango
def heom_distance(x1, x2, tipos_datos, rangos_continuos):
    distacia = 0
    for i in range(len(x1)):
        if pd.isna(x1[i]) or pd.isna(x2[i]): #Overlap
            distacia += 1
        else:
            if tipos_datos[i]:  # Categórico
                if x1[i] != x2[i]:
                    distacia += 1
            else:  # Continuo
                rango = rangos_continuos[X.columns[i]]  # Obtener el rango de la columna
                if rango == 0:  # Si el rango es 0, se trata como distancia normal
                    temp = (float(x1[i]) - float(x2[i])) ** 2
                else:
                    temp = ((float(x1[i]) - float(x2[i])) / rango) ** 2
                distacia += temp
    return np.sqrt(distacia)

# Función K-Means con HEOM
def kmeans_heom(X, tx1, k, max_iters=100):
    centroids = X.sample(n=k).values
    
    for iteration in range(max_iters):
        tx1.insert(tk.INSERT,f"\nIteración {iteration + 1}")
        clusters = [[] for _ in range(k)]  # Lista para almacenar índices de puntos
        
        for i in range(len(X)):
            point = X.iloc[i].values
            distances = [heom_distance(point, centroid, tipos_datos, rangos_continuos) for centroid in centroids]
            cluster_idx = np.argmin(distances)
            
            # Almacena el índice del punto en el clúster correspondiente
            clusters[cluster_idx].append(i)
        
        new_centroids = []
        for cluster_idx, cluster_indices in enumerate(clusters):
            if len(cluster_indices) == 0:
                continue
            
            cluster_df = X.iloc[cluster_indices]  # Obtener DataFrame de los puntos en el clúster
            new_centroid = []
            
            for col_idx, es_categorico in enumerate(tipos_datos):
                if col_idx >= cluster_df.shape[1]:
                    continue
                
                columna = cluster_df.iloc[:, col_idx]
                
                if es_categorico:  # Categórico
                    valores_categoricos = [valor for valor in columna if valor != "?"]
                    promedio_categórico = max(set(valores_categoricos), key=valores_categoricos.count) if valores_categoricos else None
                    new_centroid.append(promedio_categórico)
                else:  # Continuo
                    columna_numerica = pd.to_numeric(columna, errors='coerce')
                    promedio_continuo = columna_numerica.mean() if len(columna_numerica.dropna()) > 0 else None
                    new_centroid.append(promedio_continuo)
            
            new_centroids.append(new_centroid)
            tx1.insert(tk.INSERT,f"\nNuevo centroide para clúster {cluster_idx}: {new_centroid}\n")
        
        new_centroids = np.array(new_centroids, dtype=object)
        
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    
    tx1.insert(tk.INSERT,f"\nCentroides finales:")
    for idx, centroid in enumerate(centroids):
        tx1.insert(tk.INSERT,f"\nCentroide {idx}: {centroid}\n")
    
    return clusters, centroids



def matriz_confusion(y_real, y_pred, etiquetas=None):
    if etiquetas is None:
        etiquetas = sorted(set(y_real) | set(y_pred))
    
    matriz = {etiqueta_real: {etiqueta_pred: 0 for etiqueta_pred in etiquetas} for etiqueta_real in etiquetas}
    
    for real, pred in zip(y_real, y_pred):
        matriz[real][pred] += 1
    
    return matriz

def mostrar_matriz_confusion(matriz, tx1):
    etiquetas = sorted(matriz.keys())
    encabezado = "   " + " ".join(f"{etiqueta:>25}" for etiqueta in etiquetas)
    #tx3.insert(tk.INSERT, f"\nK vecinos mas cercanos: \n{solo_numeros}\n")
    tx1.insert(tk.INSERT, encabezado)
    #print(encabezado)
    
    for etiqueta_real in etiquetas:
        fila = [f"{matriz[etiqueta_real][etiqueta_pred]:>25}" for etiqueta_pred in etiquetas]
        tx1.insert(tk.INSERT,f"\n{etiqueta_real:>25} " + " ".join(fila))
        #print(f"{etiqueta_real:>25} " + " ".join(fila))

def calcular_metricas(matriz,):
    TP = matriz['recurrence-events']['recurrence-events']
    FP = matriz['no-recurrence-events']['recurrence-events']
    FN = matriz['recurrence-events']['no-recurrence-events']
    TN = matriz['no-recurrence-events']['no-recurrence-events']
    
    sensibilidad = TP / (TP + FN) if (TP + FN) != 0 else 0
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    exactitud = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0
    tasa_error = (FP + FN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0
    
    return {
        'Sensibilidad': sensibilidad,
        'Precisión': precision,
        'Exactitud': exactitud,
        'Tasa de Error': tasa_error
    }

def evaluaciones(tx1, clusters):
    Y = breast_cancer.data.targets
    Y = Y['Class'].tolist() #Convierte la clase en una lista unidimensional para su manejo
    y_pred = []
    etiquetas_asignadas = set()
    for cluster in clusters:
        # Obtener las etiquetas reales de los puntos en el clúster usando los índices guardados
        etiquetas_reales_cluster = [Y[idx] for idx in cluster]  # Acceder directamente a Y usando los índices
        #filtra la lista resultante de la primera línea, eliminando las etiquetas que ya han sido asignadas en otros clústeres.
        etiquetas_reales_cluster = [etiqueta for etiqueta in etiquetas_reales_cluster if etiqueta not in etiquetas_asignadas]

        if not etiquetas_reales_cluster:  # Si no quedan etiquetas no asignadas, continuar
            continue

        # Asignar la etiqueta mayoritaria
        etiqueta_pred = max(set(etiquetas_reales_cluster), key=etiquetas_reales_cluster.count)
        
        # Marcar la etiqueta como asignada
        etiquetas_asignadas.add(etiqueta_pred)
        
        # Asignar la etiqueta predicha a todos los puntos del clúster
        y_pred.extend([etiqueta_pred] * len(cluster))
    
    # Imprimir la matriz de confusión
    matriz = matriz_confusion(Y, y_pred)
    
    tx1.insert(tk.INSERT, f"\nMatriz de Confusión:")
    #print("Matriz de Confusión:")
    mostrar_matriz_confusion(matriz, tx1)
    
    metricas = calcular_metricas(matriz)
    return metricas
    

def Guardar_evaluaciones(resultados,nombre_archivo):
    df_resultados = pd.DataFrame(resultados)
    promedios = df_resultados.mean().to_dict()
    df_resultados = df_resultados._append(promedios, ignore_index=True)
    df_resultados.to_excel(nombre_archivo, index=False)

def Guardar_excel(X, indices,nombre_archivo):
    X = pd.DataFrame(X)
    # Crea un objeto ExcelWriter para guardar múltiples hojas
    with pd.ExcelWriter(nombre_archivo) as writer:
        for i, idx_cluster in enumerate(indices):
            # Verifica que haya índices para el cluster
            if idx_cluster:  # Esto asegura que no está vacío
                df_cluster = X.iloc[idx_cluster]  # Selecciona los objetos del cluster
                df_cluster.to_excel(writer, sheet_name=f'Cluster_{i+1}', index=False)
            else:
                print(f"Cluster {i+1} está vacío y no se guardará.")
