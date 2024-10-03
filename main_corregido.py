from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np

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

print(kmeans_heom(X))
