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

a=X.iloc[0]
b=X.iloc[1]
print(a)
print(b)
print(f"Distancia Final: {heom_distance(a,b,tipos_datos)}")