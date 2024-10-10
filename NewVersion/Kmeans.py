from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
from openpyxl import Workbook
from tkinter import *
import tkinter as tk

# fetch dataset 
breast_cancer = fetch_ucirepo(id=14) 
  
# data (as pandas dataframes) 
X = breast_cancer.data.features
Y = breast_cancer.data.targets


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

#Coloque tx en heom_distance y kmeans_heom, no se si dejarlo o quitarlo
def heom_distance(x1, x2, tipos_datos):
    dist = 0
    faltantes = 0
    for i in range(len(x1)):
        if pd.isna(x1[i]) or pd.isna(x2[i]):
            dist += 1
            faltantes += 1
        else:
            if tipos_datos[i]:  # Categórico
                if x1[i] != x2[i]:
                    dist += 1
            else:  # Continuo
                try: 
                    temp = (float(x1[i]) - float(x2[i])) ** 2
                    dist += temp  
                except ValueError:
                    dist += 1 if x1[i] != x2[i] else 0
    #tx1.insert(tk.INSERT,f"\nTotal de valores faltantes en esta instancia: {faltantes}")
    #print(f"Total de valores faltantes en esta instancia: {faltantes}")
    return np.sqrt(dist)

def kmeans_heom(X, tx1, k, max_iters=100):
    centroids = X.sample(n=k).values
    
    for iteration in range(max_iters):
        tx1.insert(tk.INSERT,f"\nIteración {iteration + 1}")
        #print(f"\nIteración {iteration + 1}")
        
        clusters = [[] for _ in range(k)]  # Lista para almacenar índices de puntos
        
        for i in range(len(X)):
            point = X.iloc[i].values
            distances = [heom_distance(point, centroid, tipos_datos) for centroid in centroids]
            cluster_idx = np.argmin(distances)
            
            # Almacena el índice del punto en el clúster correspondiente
            clusters[cluster_idx].append(i)
            #print(f"Punto {i} asignado al clúster {cluster_idx} con distancia {distances[cluster_idx]}")
        
        new_centroids = []
        for cluster_idx, cluster_indices in enumerate(clusters):
            if len(cluster_indices) == 0:
                continue
            
            cluster_df = X.iloc[cluster_indices]  # Obtener DataFrame de los puntos en el clúster
            new_centroid = []
            
            for col_idx, is_categorical in enumerate(tipos_datos):
                if col_idx >= cluster_df.shape[1]:
                    continue
                
                columna = cluster_df.iloc[:, col_idx]
                
                if is_categorical:  # Categórico
                    valores_categoricos = [valor for valor in columna if valor != "?"]
                    promedio_categórico = max(set(valores_categoricos), key=valores_categoricos.count) if valores_categoricos else None
                    new_centroid.append(promedio_categórico)
                else:  # Continuo
                    columna_numerica = pd.to_numeric(columna, errors='coerce')
                    promedio_continuo = columna_numerica.mean() if len(columna_numerica.dropna()) > 0 else None
                    new_centroid.append(promedio_continuo)
            
            new_centroids.append(new_centroid)
            tx1.insert(tk.INSERT,f"\nNuevo centroide para clúster {cluster_idx}: {new_centroid}\n")
            #print(f"Nuevo centroide para clúster {cluster_idx}: {new_centroid}")
        
        new_centroids = np.array(new_centroids, dtype=object)
        
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    
    #for cluster_idx, cluster_indices in enumerate(clusters):
        #print(f"\nClúster {cluster_idx}: contiene {len(cluster_indices)} puntos (índices: {cluster_indices})")
    
    tx1.insert(tk.INSERT,f"\nCentroides finales:")
    #print("\nCentroides finales:")
    for idx, centroid in enumerate(centroids):
        tx1.insert(tk.INSERT,f"\nCentroide {idx}: {centroid}\n")
        #print(f"Centroide {idx}: {centroid}")
    
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

def evaluaciones(Y,tx1, clusters):
    
    Y = Y['Class'].tolist()
    y_pred = []
    etiquetas_asignadas = set()

    for cluster in clusters:

        # Obtener las etiquetas reales de los puntos en el clúster usando los índices guardados
        etiquetas_reales_cluster = [Y[idx] for idx in cluster]  # Acceder directamente a Y usando los índices
        
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

def Guardar_excel(X, Y, indices,nombre_archivo):
    X = pd.DataFrame(X)
    Y = pd.DataFrame(Y)

    # Crea un objeto ExcelWriter para guardar múltiples hojas
    with pd.ExcelWriter(nombre_archivo) as writer:
        for i, idx_cluster in enumerate(indices):
            # Verifica que haya índices para el cluster
            if idx_cluster:  # Esto asegura que no está vacío
                df_cluster = X.iloc[idx_cluster].copy()  # Selecciona los objetos del cluster
                df_cluster['Class'] = Y.iloc[idx_cluster].squeeze() 
                df_cluster.to_excel(writer, sheet_name=f'Cluster_{i+1}', index=False)
            else:
                print(f"Cluster {i+1} está vacío y no se guardará.")

"""
def Conseguir_objetos(X, indice):
    objetos_generados=[]
    X=pd.DataFrame(X)
    #print(f"Índice: {indice}, Tipo: {type(indice)}")
    for i in indice:
        for idx in i:
        #objeto_nuevo=pd.DataFrame(columns=['Columna 1','Columna 2','Columna 3','Columna 4','Columna 5','Columna 6','Columna 7','Columna 8','Columna 9'])
            #print(f"Índice: {idx}, Tipo: {type(idx)}")
            objeto_nuevo = X.iloc[idx]
            objetos_generados.append(objeto_nuevo)
    return [pd.concat(objetos_generados, axis=0)]

# Ejecutar k-means para probar el clustering

clusters, centroids = kmeans_heom(X, k=2, max_iters=10)

# con Y a una lista de etiquetas
Y = Y['Class'].tolist()

# Ejecutar las simulaciones con las etiquetas de Y
"""