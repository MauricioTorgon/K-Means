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

# Imprimir el DataFrame para verificar las columnas
print("Datos X:")
print(X.head())  # Muestra las primeras filas de X
print("Columnas en X:")
print(X.columns)  # Muestra las columnas disponibles en X

# Definir si los atributos son categóricos o continuos
tipos_datos = {
    'class': True,
    'age': True,
    'menopause': True,
    'tumor_size': True,  # En realidad son rangos categóricos
    'inv_nodes': True,   # También son rangos categóricos
    'node_caps': True,
    'deg_malig': False,  # Continua
    'breast': True,
    'breast_quad': True,
    'irradiat': True
}

# Convertir las columnas categóricas a tipo 'category'
for col in tipos_datos.keys():
    if tipos_datos[col]:  # Si es categórico
        X[col] = X[col].astype('category')

# Asegurarse de que la columna continua ('deg_malig') es numérica
X['deg_malig'] = pd.to_numeric(X['deg_malig'], errors='coerce')

def heom_distance(x1, x2, tipos_datos):
    dist = 0
    faltantes = 0
    
    for i in range(len(x1)):
        print(f"Procesando columna {i}: {x1[i]} vs {x2[i]}")  # Mensaje para mostrar la columna y los valores
        if pd.isna(x1[i]) or pd.isna(x2[i]):
            dist += 1
            faltantes += 1
        else:
            if tipos_datos[list(tipos_datos.keys())[i]]:  # Categórico
                if x1[i] != x2[i]:
                    dist += 1
            else:  # Continua
                if i == 6:  # Solo la columna 'deg_malig' es continua
                    try:
                        # Asegurarse de que los valores sean numéricos
                        val1 = float(x1[i])
                        val2 = float(x2[i])
                        rango = X['deg_malig'].max() - X['deg_malig'].min()
                        if rango != 0:
                            temp = ((val1 - val2) / rango) ** 2
                            dist += temp
                        else:
                            dist += 1 if val1 != val2 else 0
                    except ValueError as e:
                        print(f"Error en los valores: {x1[i]}, {x2[i]}, columna 'deg_malig', tipo esperado: continuo. Detalles: {e}")
                        dist += 1  # Penalizar el error
    print(f"Total de valores faltantes en esta instancia: {faltantes}")
    return np.sqrt(dist)

def kmeans_heom(X, k=2, max_iters=100):
    # Usar dos objetos aleatorios como centroides iniciales
    random_indices = random.sample(range(len(X)), k)
    centroids = X.iloc[random_indices].values
    
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
        for cluster_id, cluster in enumerate(clusters):
            if len(cluster) > 0:
                new_centroid = []
                for j in range(len(cluster[0])):
                    if tipos_datos[list(tipos_datos.keys())[j]]:  # Categórico
                        new_centroid.append(pd.Series([c[j] for c in cluster]).mode()[0])  # Usar el modo
                    else:  # Continua
                        new_centroid.append(np.mean([c[j] for c in cluster if isinstance(c[j], (int, float))]))  # Usar la media
                new_centroids.append(new_centroid)
                print(f"Nuevo centroide {cluster_id}: {new_centroid}")
            else:
                new_centroids.append(centroids[cluster_id])
                print(f"Centroide {cluster_id} no cambió")
        
        centroids = np.array(new_centroids)
        
    return clusters, centroids

def matriz_confusion(Y_real, Y_pred):
    etiquetas = ['recurrence-events', 'no-recurrence-events']
    matriz = {etiqueta: {etiqueta_pred: 0 for etiqueta_pred in etiquetas} for etiqueta in etiquetas}
    
    for real, pred in zip(Y_real, Y_pred):
        matriz[real][pred] += 1
    
    return matriz

def mostrar_matriz_confusion(matriz):
    etiquetas = ['recurrence-events', 'no-recurrence-events']
    encabezado = f"{'':>25} {' '.join([f'{etiqueta:>25}' for etiqueta in etiquetas])}"
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

def simular_evaluaciones(Y, clusters, num_ejecuciones=20):
    resultados = []
    
    for i in range(num_ejecuciones):
        y_pred = []
        for cluster in clusters:
            etiqueta_pred = 'recurrence-events' if len(cluster) > len(clusters[0]) / 2 else 'no-recurrence-events'
            y_pred.extend([etiqueta_pred] * len(cluster))
        
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

# Ejecutar K-means
clusters, centroids = kmeans_heom(X, k=2, max_iters=10)

# Convertir a lista para las evaluaciones
Y = Y.tolist()
simular_evaluaciones(Y, clusters)
