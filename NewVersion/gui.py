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
        Evaluacion_totales = []
        self.tx1.config(state="normal")
        self.tx1.delete("0.1", tk.END)
        if self.cbx1.get() == 'K=2':
            # Ejecutar k-means para probar el clustering
            self.tx1.insert(tk.INSERT,f"\nEjecutando simulación:")
            for i in range(interaciones):
                clusters, centroids = kmeans_heom(X,tx1=self.tx1, k=2, max_iters=10)
                metricas=evaluaciones(tx1=self.tx1, clusters=clusters)
                Evaluacion_totales.append(metricas)
            
            Guardar_excel(Evaluacion_totales,nombre_archivo="evaluacion.xlsx")
            # Ejecutar las simulaciones con las etiquetas de Y
            
            #X.info