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

# 4. Definir si los atributos son categóricos o continuos
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
    """
    Calcula la distancia HEOM entre dos instancias (x1 y x2).
    tipos_datos es una lista que indica si una columna es categórica (True) o continua (False).
    """
    dist = 0
    for i in range(len(x1)):
        # Si alguna de las instancias tiene un valor faltante, sumamos 1
        if pd.isna(x1[i]) or pd.isna(x2[i]):
            dist += 1
            print("\nValor faltante.")
        else:
            if tipos_datos[i]:  # Categórico
                if x1[i] != x2[i]:
                    dist += 1
                    #print(f"\nDistancia categorica de {x1[i]} a {x2[i]} = {1}") #DEL
                #else:                                                                  #DEL
                    #print(f"\nDistancia categorica de {x1[i]} a {x2[i]} = {0}") #DEL 
            else:  # Continuo
                try: 
                    temp= (float(x1[i]) - float(x2[i])) ** 2 #/ max_range[i] ** 2# Aseguramos que las variables continuas sean numéricas
                    dist += temp  
                   # print(f"\nDistancia continua de {x1[i]} a {x2[i]} = {temp}")  #DEL
                except ValueError:
                    dist += 1 if x1[i] != x2[i] else 0
    return np.sqrt(dist)

# 5. Implementación del K-Means con HEOM sin modificar datos
def kmeans_heom(X, k=2, max_iters=100):
    # Elegir dos instancias aleatorias como centroides iniciales
    centroids = X.sample(n=k).values
    
    for _ in range(max_iters):
        clusters = [[] for _ in range(k)]
        
        # Asignar cada punto al clúster más cercano según HEOM
        for i, point in X.iterrows():
            distances = [heom_distance(point, centroid, tipos_datos)
                         for centroid in centroids]
            cluster_idx = np.argmin(distances)
            clusters[cluster_idx].append(point)
        
        # Recalcular centroides
        new_centroids = []
        for cluster in clusters:
            new_centroid = []
            if len(cluster) == 0:
                continue  # Evitar recalcular para clusters vacíos
            
            cluster_df = pd.DataFrame(cluster)  # Convertimos el cluster a un DataFrame

            for col_idx, is_categorical in enumerate(tipos_datos):
                if col_idx >= cluster_df.shape[1]:
                    continue  # Evitar el acceso fuera de los límites

                columna = cluster_df.iloc[:, col_idx]
                
                if is_categorical:  # Si es categórica
                    # Filtramos valores no válidos como "?"
                    valores_categoricos = [valor for valor in columna if valor != "?"]
                    
                    # Calculamos el "promedio categórico", es decir, el valor más frecuente
                    promedio_categorico = max(set(valores_categoricos), key=valores_categoricos.count) if valores_categoricos else None
                    new_centroid.append(promedio_categorico)
                else:  # Si es continua
                    # Intentamos convertir la columna a numérico (ignorando errores)
                    columna_numerica = pd.to_numeric(columna, errors='coerce')
                    
                    # Calculamos la media de los valores numéricos
                    promedio_continuo = columna_numerica.mean() if len(columna_numerica.dropna()) > 0 else None
                    new_centroid.append(promedio_continuo)

            new_centroids.append(new_centroid)
        
        new_centroids = np.array(new_centroids, dtype=object)  # Asegurarse de que los tipos se manejen correctamente
        
        # Si los centroides no cambian, se detiene
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    
    return clusters, centroids

def matriz_confusion(y_real, y_pred, etiquetas=None):
    """
    Crea una matriz de confusión a partir de los valores reales y predichos.
    
    :param y_real: Lista o serie de valores reales (etiquetas verdaderas).
    :param y_pred: Lista o serie de valores predichos por el modelo.
    :param etiquetas: Lista de etiquetas (opcional). Si no se proporciona, se infiere de los datos.
    :return: Matriz de confusión como un diccionario de diccionarios.
    """
    # Si no se proporcionan etiquetas, inferirlas de los datos
    if etiquetas is None:
        etiquetas = sorted(set(y_real) | set(y_pred))
    
    # Inicializar la matriz de confusión como un diccionario de diccionarios
    matriz = {etiqueta_real: {etiqueta_pred: 0 for etiqueta_pred in etiquetas} for etiqueta_real in etiquetas}
    
    # Rellenar la matriz de confusión
    for real, pred in zip(y_real, y_pred):
        matriz[real][pred] += 1
    
    return matriz

# Función para mostrar la matriz de confusión de forma más legible
def mostrar_matriz_confusion(matriz):
    etiquetas = sorted(matriz.keys())
    encabezado = "   " + " ".join(f"{etiqueta:>25}" for etiqueta in etiquetas)
    print(encabezado)
    
    for etiqueta_real in etiquetas:
        fila = [f"{matriz[etiqueta_real][etiqueta_pred]:>25}" for etiqueta_pred in etiquetas]
        print(f"{etiqueta_real:>25} " + " ".join(fila))


def calcular_metricas(matriz):
    """
    Calcula sensibilidad, exactitud, precisión y tasa de error a partir de una matriz de confusión.
    matriz: Diccionario de diccionarios (matriz de confusión).
    Retorna un diccionario con las métricas.
    """
    # Verdaderos positivos (TP), Falsos positivos (FP), Verdaderos negativos (TN), Falsos negativos (FN)
    TP = matriz['recurrence-events']['recurrence-events']
    FP = matriz['no-recurrence-events']['recurrence-events']
    FN = matriz['recurrence-events']['no-recurrence-events']
    TN = matriz['no-recurrence-events']['no-recurrence-events']
    
    # Calcular las métricas
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
    """
    Simula 20 ejecuciones, calcula las métricas y las guarda en un archivo Excel.
    Y: Etiquetas reales de clase.
    num_ejecuciones: Número de ejecuciones a realizar (por defecto 20).
    """
    # Crear un DataFrame para almacenar los resultados
    resultados = []
    
    for i in range(num_ejecuciones):
        # Simulamos las predicciones aleatorias (puedes sustituirlo con un modelo real)
        y_pred = [random.choice(['recurrence-events', 'no-recurrence-events']) for _ in range(len(Y))]
        
        # Crear la matriz de confusión
        matriz = matriz_confusion(Y, y_pred)
        
        # Calcular las métricas para esta ejecución
        metricas = calcular_metricas(matriz)
        resultados.append(metricas)
    
    # Convertir los resultados en DataFrame
    df_resultados = pd.DataFrame(resultados)
    
    # Calcular los promedios de cada métrica
    promedios = df_resultados.mean().to_dict()
    
    # Añadir una fila de promedios al final
    promedios['Sensibilidad'] = 'Promedio'  # Colocamos "Promedio" solo en una columna de texto
    df_resultados = df_resultados._append(promedios, ignore_index=True)
    
    # Guardar los resultados en un archivo Excel
    df_resultados.to_excel("evaluacion.xlsx", index=False)

# Asumiendo que Y es una lista de etiquetas
Y = Y['Class'].tolist()  # Asegúrate de que Y sea la columna de clases

# Simular las evaluaciones y guardar el archivo
simular_evaluaciones(Y)

""""
# Ejemplo de uso
Y = Y['Class'].tolist()  # Extraer la columna 'Class' como una lista
y_pred = ['no-recurrence-events'] * 200 + ['recurrence-events'] * 86  # Simulación de predicciones

matriz = matriz_confusion(Y, y_pred)
mostrar_matriz_confusion(matriz)

# 5. Evaluación del modelo y ejecución 20 veces
results = []

for _ in range(20):
    clusters, centroids = kmeans_heom(X)
    results.append((clusters,centroids))
    
# 6. Guardar los resultados en Excel
df_results.to_excel('evaluacion.xlsx', index=False)
"""