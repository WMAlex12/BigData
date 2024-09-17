import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

#Cargar datos 

datos = np.array([[0.19, 0.5, 0.4], 
                 [0.28, 0.8, 0.6], 
                 [0.3, 0.9, 0.7],
                 [0.25, 1.1, 1.2], 
                 [0.29, 1.3, 1.4],
                 [0.28, 1.4, 1.7],
                 [0.5, 0.8, 1.4]])

print(datos.ndim)

Y = datos[:,0] #Variable dependiente 
x1 = datos[:,1] # var. independiente
x2 = datos[:,2] # var. independiente 

Unos = np.ones((len(x1),1))
X = np.append(Unos, datos[:,1:3], axis=1)


#Etapa de aprendizaje 
A = np.matmul(np.transpose(X), X)
K = np.linalg.inv(A)
## eL @ Sirve para multiplicar matrices y vectores 
B = K @ np.transpose(X) @ Y ##


#Visualizacion 
plano3D = plt.figure()
ax = plano3D.add_subplot(projection='3d')
ax.scatter(x1,x2,Y,color="red")
ax.grid()

#Generar plano aprendido 

x = np.linspace(x1.min(), x1.max(),10)
y = np.linspace(x2.min(), x2.max(),10)
[XFIT, YFIT] = np.meshgrid(x,y)

##Modelo aprendido 
ZFIT = B[0]+B[1]*XFIT+B[2]*YFIT
ax.plot_surface(XFIT,YFIT,ZFIT, cmap = 'viridis') #Se imprime el plano 


plt.show()

##Analisisd de residuos 
M = B[0] + B[1]*x1 +B[2]*x2 #Modelo aprendido
Error = M - Y
fig, (ax1, ax2) = plt.subplots(2,1,figsize = (8,5))
ax1.grid()
x_int = np.linspace(0,len(Error), len( Error))
ax1.plot(x_int, Error)
ax2.hist(Error)
ax2.grid()

plt.show()


### Metodo de persistencia 

sp = pd.read_excel("/home/karelvazquez/Documents/WMAlex/BigData/Manejo_de_datos/StockPrice.xlsx", sheet_name="StockPrice")
TSLA = sp.loc[0:49,"TESLA"]

fig, ax = plt.subplots(figsize = (7,5))
x_int = [x for x in range(0,len(TSLA))] ###Comprension de lista 
ax.plot(x_int, TSLA)
ax.grid()
Persistencia  = sp.loc[1:50, "TESLA"]
ax.plot(x_int, Persistencia, color = "red")

plt.show()
