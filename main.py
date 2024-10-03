from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
import openpyxl

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
        if pd.isna(x1.iloc[i]) or pd.isna(x2[i]):
            dist += 1
            print("\nValor faltante.")
        else:
            if tipos_datos[i]:  # Categórico
                if x1.iloc[i] != x2.iloc[i]:
                    dist += 1
                    print(f"\nDistancia categorica de {x1.iloc[i]} a {x2.iloc[i]} = {1}") #DEL
                else:                                                                     #DEL
                    print(f"\nDistancia categorica de {x1.iloc[i]} a {x2.iloc[i]} = {0}") #DEL 
            else:  # Continuo
                try: 
                    temp= (float(x1.iloc[i]) - float(x2.iloc[i])) ** 2 # Aseguramos que las variables continuas sean numéricas
                    dist += temp  
                    print(f"\nDistancia continua de {x1.iloc[i]} a {x2.iloc[i]} = {temp}")  #DEL
                except ValueError:
                    dist += 1 if x1.iloc[i] != x2.iloc[i] else 0
    return np.sqrt(dist)

# 5. Implementación del K-Means con HEOM sin modificar datos
# 4. Implementación del K-Means con HEOM
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
            new_centroids.append(np.mean(cluster, axis=0))
        new_centroids = np.array(new_centroids)
        
        # Si los centroides no cambian, se detiene
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    
    return clusters, centroids

# 5. Evaluación del modelo y ejecución 20 veces
results = []

for _ in range(20):
    clusters, centroids = kmeans_heom(X)
    results.append(clusters,centroids)
    
# 6. Guardar los resultados en Excel
df_results.to_excel('evaluacion.xlsx', index=False)