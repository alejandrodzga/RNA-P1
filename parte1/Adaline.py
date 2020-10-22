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

    #print("CURIOSIDAD ",filadatos[:8])

    matrizIncrementoPesos = (razon*(resultadoEsperado-resultado))*filadatos[:8]

    #print("matrizincremetopesos ",matrizIncrementoPesos)
    matrizNuevosPesos = matrizpesos + matrizIncrementoPesos

    return matrizNuevosPesos

# Funcion para calcular el nuevo umbral en funcion de la salida obtenida
def calcNuevoUmbral(razon, resultado, resultadoEsperado,umbral):
    
    incrementoUmbral = razon*(resultadoEsperado-resultado)
    nuevoUmbral = umbral + incrementoUmbral
    return nuevoUmbral


# Funcion Error Cuadratico medio MSE 
def calcMSE(columnadatos,matrizsalidas,numerofilas):
    
    #print("COLUMNADATOS: ",columnadatos)
    #print("MATRIZSALIDAS: ",matrizsalidas)

    np.array(columnadatos)
    np.array(matrizsalidas)
  
    matrizdiff = np.subtract(columnadatos,matrizsalidas)

    #print("MATRIZdiff: ",matrizdiff)
    matrizdiff = matrizdiff**2

    #print("MATRIZdiff SQUAREE ",matrizdiff)
    resultadoN = np.sum(matrizdiff)
    #print("RESULTADO N ",resultadoN)
    resultadoN = resultadoN/numerofilas
    #print("Resultado del ERROR MSE ", resultadoN)
    return resultadoN




# Funcion Error absoluto medio
def calcMAE(columnadatos,matrizsalidas,numerofilas):
    
    #print("COLUMNADATOS: ",columnadatos)
    #print("MATRIZSALIDAS: ",matrizsalidas)

    np.array(columnadatos)
    np.array(matrizsalidas)
  
    matrizdiff = np.subtract(columnadatos,matrizsalidas)

    #print("MATRIZdiff: ",matrizdiff)
    matrizdiff = abs(matrizdiff)

    #print("MATRIZdiff SQUAREE ",matrizdiff)
    resultadoN = np.sum(matrizdiff)
    #print("RESULTADO N ",resultadoN)
    resultadoN = resultadoN/numerofilas
    #print("Resultado del ERROR MAE ", resultadoN)
    return resultadoN





#######################################################  MAIN   ############################################# 

# Inicializamos resultado
resultado = 0

# Numero de filas de la matriz de entrenamiento
trainRows = len(trainData)

matrizSalidasTrain = np.empty(trainRows)

#np.array(matrizSalidasTrain)

# Numero de filas de la matriz de validacion
validationRows = len(validationData)

matrizSalidasValidation = np.empty(validationRows)

# Numero de filas de la matriz de test
testRows = len(testData)

matrizSalidasTest = np.empty(testRows)



# Aqui guardamos cada resultado con el objetivo de tener todas las salidas para el calculo de error

# MATRICES PARA GUARDAR TODOS LOS ERRORES DE CADA ITERACION Y POSTERIORMENTE HACER GRAFICAS...


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
        #print("matriz de salida ",j,"valor:",matrizSalidasTrain[j])
        
        
    
        # Actualizamos los pesos 
        pesos = calcNuevosPesos(pesos, trainData[j], razon, trainData[j][8], matrizSalidasTrain[j])
        umbral = calcNuevoUmbral(razon,matrizSalidasTrain[j],trainData[j][8],umbral) 
        #print("matriz pesos ",pesos)
        #print(" umbral ",umbral)
    for k in range(validationRows):
        matrizSalidasValidation[k] = calcSalida(pesos,validationData[k],umbral)


    #print(matrizSalidasTrain[:])
    #print("TRASDFASFDADSF",trainData[:,8])
    # Calculamos los errores
    matrizErroresCuadraticos[i] = calcMSE(trainData[:,8],matrizSalidasTrain[:],trainRows)
    matrizErroresAbsolutos[i] = calcMAE(trainData[:,8],matrizSalidasTrain[:],trainRows)
    
    VmatrizErroresCuadraticos[i] = calcMSE(validationData[:,8],matrizSalidasValidation[:],validationRows)
    VmatrizErroresAbsolutos[i] = calcMAE(validationData[:,8],matrizSalidasValidation[:],validationRows)
    
    print("Resultados de MSE: ",matrizErroresCuadraticos[i])
    print("Resultados de MAE: ",matrizErroresAbsolutos[i])




# TODO HACER UNA RECTA EN LUGAR DE UN BUCLE...
for o in range(nciclos):
    for l in range(testRows):
        matrizSalidasTest[l] = calcSalida(pesos,testData[l],umbral)

    TmatrizErroresAbsolutos[o] = calcMAE(testData[:,8],matrizSalidasTest[:],testRows)
    TmatrizErroresCuadraticos[o] = calcMSE(testData[:,8],matrizSalidasTest[:],testRows)
    
    print("Resultados de MSE TEST: ",TmatrizErroresCuadraticos[o])
    print("Resultados de MAE TEST: ",TmatrizErroresAbsolutos[o])



# x axis values 
x = np.arange(0,nciclos)
# corresponding y axis values 

  
# plotting the points  
#plt.plot(x,log(matrizErroresAbsolutos))
#plt.legend(handles=[], loc='upper right')


# TODO VER LA LEYENDA Y VER SI HAY QUE PONER ERRORES EN ESCALA LOGARITMICA O NO
plt.plot(x,VmatrizErroresCuadraticos,label="Error validacion")
plt.plot(x,matrizErroresCuadraticos,label="Error entrenamiento") 
plt.plot(x,TmatrizErroresCuadraticos,label="Error test") 

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
# naming the x axis 
plt.xlabel('ciclos') 
# naming the y axis 
plt.ylabel('Valor de Error') 
  
# giving a title to my graph 
plt.title('My first graph!') 

# function to show the plot 
plt.show() 

################################################################ FIN MAIN ##########################################################################
"""
PRUEBAS FUNCIONAMIENTO INCREMENTO PESOS Y UMBRAL
resultado = calcSalida(pesos,trainData[0],umbral)
nuevosPesos = calcNuevosPesos(pesos, trainData[0], razon, trainData[0][8], resultado)
print(trainData[0])

print("resultado y FINAL: ",resultado)
print("pesos FINALES CALCULADOS: ",pesos)
print("UMBRAL FINAL: ",umbral)
"""



# FIXME PRUEBA DEL METODO MSE
"""
matrizA = np.array([[1],[2],[3],[4]])
matrizB = np.array([[1],[2],[3],[3.9]])

resab = calcMAE(matrizA,matrizB,4)
print("TRESTTT ",resab)

resabA = calcMSE(matrizA,matrizB,4)
print("TRESTTT ",resabA)

"""


# TODO VER QUE TIPO DE ERROR HAY QUE USAR EN ENTRENAMIENTO Y VALIDACION
# TODO VER COMO FUNCIONAN LOS CONJUNTOS DE TEST Y VALIDACION Y CUANDO HAY QUE USARLOS 
# TODO GRAFICAS CON LOS ERRORES DE TODAS LAS ITERACIONES