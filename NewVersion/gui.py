from tkinter import Tk, Button, Label, Entry, Frame, Text
from tkinter.ttk  import Combobox
from tkinter.filedialog import askopenfilename
import tkinter as tk
import pathlib
from Kmeans import *

class gui:
    values_Data = ['K=2', 'K=4 y k=6']
    file = 0
    
    def __init__(self):
        self.path = ""

        """ --- Creacion y configuracion de la GUI --- """
        #CREACION DE VENTANA TK
        self.mainFrame = Tk()

        # --- Configuracion de TK ---
        self.mainFrame.resizable(0,0) #no se puede cambiar el tamaño de la ventana
        self.mainFrame.geometry("1000x600")#Establecemos el tamaño predeterminado de la ventana
        self.mainFrame.title("Knn")

        self.fr1 = Frame(self.mainFrame)
        self.fr1.pack(expand=True, fill='both')

        self.lb1 = Label(self.fr1, text="No se que poner aquí:")
        self.lb1.place(x=15, y=20)
        self.cbx1 = Combobox(self.fr1, width=5, values=self.values_Data, state="readonly")
        self.cbx1.place(x=150, y=20, width= 150)

        # -> Creamos el boton de Knn
        self.bt1 = Button(self.fr1, text="Kmeans", command=self.Kmeans) #command=self.Kmeans
        self.bt1.place(x=880, y=560, width=100, height=25)

        self.tx1 = Text(self.fr1, state="disabled")
        self.tx1.place(x=15, y=50, width=970, height=500)

        self.mainFrame.mainloop()

    def Kmeans(self):
        interaciones = 20
        self.tx1.config(state="normal")
        self.tx1.delete("0.1", tk.END)
        self.tx1.insert(tk.INSERT,f"\nEjecutando K-Means:")
        if self.cbx1.get() == 'K=2':
            Evaluacion_totales = []
            # Ejecutar k-means para probar el clustering
            for i in range(interaciones):
                clusters, centroids = kmeans_heom(X,tx1=self.tx1, k=2)
                metricas=evaluaciones(Y,tx1=self.tx1, clusters=clusters)
                Evaluacion_totales.append(metricas)
            
            Guardar_evaluaciones(Evaluacion_totales,nombre_archivo="evaluacion.xlsx")
        else:
            #Indice_K4=[]
            #Indice_K6=[]

            clusters_K4,centroids_K4 =kmeans_heom(X,tx1=self.tx1, k=4)

            clusters_K6,centroids_K6=kmeans_heom(X,tx1=self.tx1, k=6)
            #Indice_K6.append(clusters_K6)

            Guardar_excel(X,Y,clusters_K4, nombre_archivo="K-MEANSGROUP_K4.xlsx")
            Guardar_excel(X,Y,clusters_K6, nombre_archivo="K-MEANSGROUP_K6.xlsx")
            # Ejecutar las simulaciones con las etiquetas de Y
            
            #X.info