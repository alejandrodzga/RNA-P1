# Practica 1 de Redes de Neuronas artificiales. Programacion de Adaline. 
import numpy as np
import sys
import random
import matplotlib.pyplot as plt


# Entrada de archivos de datos, ATENCION: Estos archivos son creados al ejecutar preprocessingData.py
f1 = open('trainData.txt')
trainData = np.loadtxt(f1,dtype=float, delimiter=',',skiprows=0)
trainData = np.array(trainData)
f1.close()
    
f2 = open('testData.txt')
testData = np.loadtxt(f2,dtype=float, delimiter=',',skiprows=0)
testData = np.array(testData)
f2.close()
    
f3 = open('validationData.txt')
validationData = np.loadtxt(f3,dtype=float, delimiter=',',skiprows=0)
validationData = np.array(validationData)
f3.close()
    

# Inicializamos de forma random un vector con los pesos con valores comprendidos entre 0 y 1
# Tenemos 8 pesos y otro elemento mas dedicado al umbral, un total de 9 elementos en el vector
pesos = np.random.rand(1,8)
umbral = random.random()
# NOTA: Estamos trabajando con float64 en la matriz de pesos 

# Caso de error en el que no se le pasan los parametros necesarios 
if(len(sys.argv)!=3):
    raise TypeError("Numero incorrecto de parametros!")


# Entrada de los hiperparametros 
# El formato de entrada de parametros es el siguiente: Adaline.py nciclos razon
nciclos = int(sys.argv[1])
razon = float(sys.argv[2])



# Funcion que multiplica los pesos por las entradas y devuelve una salida real 
def calcSalida(matrizpesos, filadatos, umbral):
    # Realizamos la multiplicacion entre vectores (cada entrada con su peso correspondiente)
    matrizres = matrizpesos*filadatos[:8]
    
    # Realizamos el sumatorio de la matriz resultante al multiplicar los pesos con las entradas para obtener la salida
    salidaReal = np.sum(matrizres, axis=1)

    # Convertimos a numero
    salidaReal = salidaReal[0]+umbral 

    return salidaReal

# Funcion para el ajuste de los nuevos pesos en funcion de la salida obtenida y la entrada 
def calcNuevosPesos(matrizpesos, filadatos, razon, resultadoEsperado, resultado):
    
    # Aplicamos la formula de incremento/decremento de pesos para ajustar los pesos anteriores

    matrizIncrementoPesos = (razon*(resultadoEsperado-resultado))*filadatos[:8]

    
    matrizNuevosPesos = matrizpesos + matrizIncrementoPesos

    return matrizNuevosPesos

# Funcion para calcular el nuevo umbral en funcion de la salida obtenida  
def calcNuevoUmbral(razon, resultado, resultadoEsperado,umbral):
    
    incrementoUmbral = razon*(resultadoEsperado-resultado)
    nuevoUmbral = umbral + incrementoUmbral
    return nuevoUmbral


# Funcion Error Cuadratico medio MSE  
def calcMSE(columnadatos,matrizsalidas,numerofilas):
    
    np.array(columnadatos)
    np.array(matrizsalidas)
  
    matrizdiff = np.subtract(columnadatos,matrizsalidas)

    
    matrizdiff = matrizdiff**2

    
    resultadoN = np.sum(matrizdiff)
    
    resultadoN = resultadoN/numerofilas
    
    return resultadoN




# Funcion Error absoluto medio 
def calcMAE(columnadatos,matrizsalidas,numerofilas):
    
    np.array(columnadatos)
    np.array(matrizsalidas)
  
    matrizdiff = np.subtract(columnadatos,matrizsalidas)

    
    matrizdiff = abs(matrizdiff)

    
    resultadoN = np.sum(matrizdiff)
    
    resultadoN = resultadoN/numerofilas
    
    return resultadoN


# Funcion encargada de desnormalizar los datos para la salida
def denormalice(matrizsalidas,numerofilas,numerocolumnas):

    # Primero obtenemos los datos maximos y minimos de la columna de datos esperados
    f = open('Data/california_housing_train.dat')
    data = np.loadtxt(f,dtype=float, delimiter=',',skiprows=1)
    np.array(data)
    f.close()

    # Sacamos la columna de datos esperados
    cesperados = data[:,8]
    # obtenemos los datos maximos y minimos para la denormalizacion 
    maximo = np.amax(cesperados)
    minimo = np.amin(cesperados)

    # Formula para desnormalizar los valores
    matrizsalidas[:,1] = matrizsalidas[:,1]*(maximo-minimo)+minimo
    matrizsalidas[:,0] = matrizsalidas[:,0]*(maximo-minimo)+minimo

    return matrizsalidas


#######################################################  MAIN   ############################################# 

# Inicializamos resultado
resultado = 0

# Numero de filas de la matriz de entrenamiento
trainRows = len(trainData)

# Salidas matriz de Entrenamiento
matrizSalidasTrain = np.empty(trainRows)

#np.array(matrizSalidasTrain)

# Numero de filas de la matriz de validacion
validationRows = len(validationData)

# Salidas matriz de Validacion
matrizSalidasValidation = np.empty(validationRows)

# Numero de filas de la matriz de test
testRows = len(testData)


# Salidas matriz de Test
matrizSalidasTest = np.empty(testRows)



# MATRICES PARA GUARDAR TODOS LOS ERRORES DE CADA ITERACION Y POSTERIORMENTE HACER GRAFICAS
matrizErroresCuadraticos = np.empty(nciclos)
matrizErroresAbsolutos = np.empty(nciclos)

VmatrizErroresCuadraticos = np.empty(nciclos)
VmatrizErroresAbsolutos = np.empty(nciclos)


TmatrizErroresCuadraticos = np.empty(nciclos)
TmatrizErroresAbsolutos = np.empty(nciclos)

# Bucle for de los CICLOS 
for i in range(nciclos):
    
    # Bucle para recorrer los patrones de entrenamiento
    for j in range(trainRows):
        
        # Calculamos la salida de cada patron de entrenamiento y lo guardamos en la matriz de salidas de la iteracion
        matrizSalidasTrain[j] = calcSalida(pesos,trainData[j],umbral)
        
        # Actualizamos los pesos y umbral
        pesos = calcNuevosPesos(pesos, trainData[j], razon, trainData[j][8], matrizSalidasTrain[j])
        umbral = calcNuevoUmbral(razon,matrizSalidasTrain[j],trainData[j][8],umbral) 
        
    # Calculamos las salidas con el conjunto de validacion
    for k in range(validationRows):
        matrizSalidasValidation[k] = calcSalida(pesos,validationData[k],umbral)


    # Calculamos los errores de entrenamiento y validacion
    matrizErroresCuadraticos[i] = calcMSE(trainData[:,8],matrizSalidasTrain[:],trainRows)
    matrizErroresAbsolutos[i] = calcMAE(trainData[:,8],matrizSalidasTrain[:],trainRows)
    
    VmatrizErroresCuadraticos[i] = calcMSE(validationData[:,8],matrizSalidasValidation[:],validationRows)
    VmatrizErroresAbsolutos[i] = calcMAE(validationData[:,8],matrizSalidasValidation[:],validationRows)
    


# Errores de test
for o in range(nciclos):
    for l in range(testRows):
        matrizSalidasTest[l] = calcSalida(pesos,testData[l],umbral)

    # Calculamos los errores de test
    TmatrizErroresAbsolutos[o] = calcMAE(testData[:,8],matrizSalidasTest[:],testRows)
    TmatrizErroresCuadraticos[o] = calcMSE(testData[:,8],matrizSalidasTest[:],testRows)
    


# Imprimimos los ultimos errores:

# Errores finales conjunto de entrenamiento
print("Ultimo MSE entrenamiento: ",matrizErroresCuadraticos[nciclos-1])
print("Ultimo MAE entrenamiento: ",matrizErroresAbsolutos[nciclos-1])

# Errores finales conjunto validacion
print("Ultimo MSE validacion: ",VmatrizErroresCuadraticos[nciclos-1])
print("Ultimo MAE validacion: ",VmatrizErroresAbsolutos[nciclos-1])

# Errores finales conjunto test
print("Ultimo MSE test: ",TmatrizErroresCuadraticos[nciclos-1])
print("Ultimo MAE test: ",TmatrizErroresAbsolutos[nciclos-1])


# SALIDAS

# Salida de los datos del modelo
print("Modelo final, Pesos: ",pesos,"Umbral:",umbral)

modelo = np.append(pesos,umbral)
#print("new pesos+umbral: ",modelo)

modelfilename = 'Modelo'+str(nciclos)+'Ciclos'+str(razon)+'Razon.txt'
#'hanning(%d).pdf' % num    %razon

f1 = open(modelfilename, "w")
# Guardamos la matriz en su formato en el archivo de salida de texto
np.savetxt(f1, modelo, delimiter=' , ', fmt='%f')
f1.close()


# Salida de los errores de entrenamiento y validacion MSE por cada ciclo de entrenamiento
mserror = np.empty((nciclos,2))
mserror[:,0] = matrizErroresCuadraticos
mserror[:,1] = VmatrizErroresCuadraticos

f2 = open('ErroresMSE.txt', "w")

np.savetxt(f2, mserror, delimiter=' , ', fmt='%f')
f2.close()


maerror = np.empty((nciclos,2))
maerror[:,0] = matrizErroresAbsolutos
maerror[:,1] = VmatrizErroresAbsolutos

f3 = open('ErroresMAE.txt', "w")

np.savetxt(f3, maerror, delimiter=' , ', fmt='%f')
f3.close()



#-----------------------------------------------------------
# SALIDAS DE LOS DATOS PARA COMPARAR OBTENIDOS CON ESPERADOS

# Matriz de salidas obtenidas con el conjunto de test y salidas esperadas
cmsalida = np.zeros((testRows,2))
cmsalida[:,0] = matrizSalidasTest
cmsalida[:,1] = testData[:,8]

# Desnormalizamos los valores de las salidas
cmsalida = denormalice(cmsalida,testRows,2)

# Sacamos por archivo de texto ambas salidas, la obtenida y la esperada sin normalziar
s1filename = 'TESTsalidas'+str(nciclos)+'ciclos'+str(razon)+'razon.txt'
f2 = open(s1filename, "w")

# Guardamos la matriz en su formato en el archivo de salida de texto
np.savetxt(f2, cmsalida, delimiter=' , ', fmt='%f')
f1.close()



#-----------------------------------------------------------


# GRAFICAS DE LOS ERRORES MSE Y MAE DE LOS DIFERENTES CONJUNTOS DE DATOS

# Eje X numero de ciclos (Temporal)
x = np.arange(0,nciclos)

# Grafica error MSE
# En caso de querer en escala logaritmica las tablas de los errores descomentar
"""
plt.xscale('log')
plt.yscale('log')
"""

plt.plot(x,VmatrizErroresCuadraticos,label="Error validacion")
plt.plot(x,matrizErroresCuadraticos,label="Error entrenamiento") 
plt.plot(x,TmatrizErroresCuadraticos,'--r',label="Error test") 

plt.legend()
# naming the x axis 
plt.xlabel('Ciclos') 
# naming the y axis 
plt.ylabel('Valor Error Cuadratico Medio (MSE)') 
  
# giving a title to my graph 
plt.title('Errores MSE') 

# function to show the plot 
plt.show() 


# Grafica error MAE
# En caso de querer en escala logaritmica las tablas de los errores descomentar
"""
plt.xscale('log')
plt.yscale('log')
"""

plt.plot(x,VmatrizErroresAbsolutos,label="Error validacion")
plt.plot(x,matrizErroresAbsolutos,label="Error entrenamiento") 
plt.plot(x,TmatrizErroresAbsolutos,'--r',label="Error test") 

plt.legend()
# naming the x axis 
plt.xlabel('Ciclos') 
# naming the y axis 
plt.ylabel('Valor Error Medio Absoluto (MAE)') 
  
# giving a title to my graph 
plt.title('Errores MAE') 

# function to show the plot 
plt.show() 


################################################################ FIN MAIN ##########################################################################




