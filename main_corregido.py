from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
import random
from openpyxl import Workbook

# fetch dataset 
breast_cancer = fetch_ucirepo(id=14) 
  
# data (as pandas dataframes) 
X = breast_cancer.data.features 
Y = breast_cancer.data.targets 

# Definir si los atributos son categóricos o continuos
tipos_datos = [
    True,  # class
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
    print(f"Total de valores faltantes en esta instancia: {faltantes}")
    return np.sqrt(dist)

def kmeans_heom(X, k=2, max_iters=100):
    centroids = X.sample(n=k).values
    
    for iteration in range(max_iters):
        print(f"\nIteración {iteration + 1}")
        clusters = [[] for _ in range(k)]
        
        for i in range(len(X)):
            point = X.iloc[i].values
            distances = [heom_distance(point, centroid, tipos_datos) for centroid in centroids]
            cluster_idx = np.argmin(distances)
            
            print(f"Punto {i} asignado al clúster {cluster_idx} con distancia {distances[cluster_idx]}")
            
            clusters[cluster_idx].append(point)
        
        new_centroids = []
        for cluster_idx, cluster in enumerate(clusters):
            if len(cluster) == 0:
                continue
            
            cluster_df = pd.DataFrame(cluster)
            new_centroid = []
            
            for col_idx, is_categorical in enumerate(tipos_datos):
                if col_idx >= cluster_df.shape[1]:
                    continue
                
                columna = cluster_df.iloc[:, col_idx]
                
                if is_categorical:  # Categórico
                    valores_categoricos = [valor for valor in columna if valor != "?"]
                    promedio_categorico = max(set(valores_categoricos), key=valores_categoricos.count) if valores_categoricos else None
                    new_centroid.append(promedio_categorico)
                else:  # Continuo
                    columna_numerica = pd.to_numeric(columna, errors='coerce')
                    promedio_continuo = columna_numerica.mean() if len(columna_numerica.dropna()) > 0 else None
                    new_centroid.append(promedio_continuo)
            
            new_centroids.append(new_centroid)
            print(f"Nuevo centroide para clúster {cluster_idx}: {new_centroid}")
        
        new_centroids = np.array(new_centroids, dtype=object)
        
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    
    for cluster_idx, cluster in enumerate(clusters):
        print(f"\nClúster {cluster_idx}: contiene {len(cluster)} puntos")
    
    print("\nCentroides finales:")
    for idx, centroid in enumerate(centroids):
        print(f"Centroide {idx}: {centroid}")
    
    return clusters, centroids

def matriz_confusion(y_real, y_pred, etiquetas=None):
    if etiquetas is None:
        etiquetas = sorted(set(y_real) | set(y_pred))
    
    matriz = {etiqueta_real: {etiqueta_pred: 0 for etiqueta_pred in etiquetas} for etiqueta_real in etiquetas}
    
    for real, pred in zip(y_real, y_pred):
        matriz[real][pred] += 1
    
    return matriz

def mostrar_matriz_confusion(matriz):
    etiquetas = sorted(matriz.keys())
    encabezado = "   " + " ".join(f"{etiqueta:>25}" for etiqueta in etiquetas)
    print(encabezado)
    
    for etiqueta_real in etiquetas:
        fila = [f"{matriz[etiqueta_real][etiqueta_pred]:>25}" for etiqueta_pred in etiquetas]
        print(f"{etiqueta_real:>25} " + " ".join(fila))

def calcular_metricas(matriz):
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

def simular_evaluaciones(Y, num_ejecuciones=20):
    resultados = []
    
    for i in range(num_ejecuciones):
        y_pred = [random.choice(['recurrence-events', 'no-recurrence-events']) for _ in range(len(Y))]
        
        print(f"\nEjecutando simulación {i + 1}:")
        print(f"Predicciones: {y_pred}")
        
        matriz = matriz_confusion(Y, y_pred)
        
        print("Matriz de Confusión:")
        mostrar_matriz_confusion(matriz)
        
        metricas = calcular_metricas(matriz)
        resultados.append(metricas)
    
    df_resultados = pd.DataFrame(resultados)
    promedios = df_resultados.mean().to_dict()
    promedios['Sensibilidad'] = 'Promedio'
    df_resultados = df_resultados._append(promedios, ignore_index=True)
    
    df_resultados.to_excel("evaluacion.xlsx", index=False)

# Ejecutar k-means para probar el clustering
clusters, centroids = kmeans_heom(X, k=2, max_iters=10)

# Asumiendo que Y es una lista de etiquetas
Y = Y['Class'].tolist()

# Ejecutar las simulaciones con las etiquetas de Y
simular_evaluaciones(Y)
X.info