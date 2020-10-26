import numpy as np
import sys

f1 = open('modelo.txt')
modelo = np.loadtxt(f1,dtype=float, delimiter=',',skiprows=0)
modelo = np.array(modelo)
f1.close()

pesos = modelo[:8]

umbral = modelo[8]


print("pesos: ", pesos)
print("umbral: ",umbral)

f2 = open('testData.txt')
testData = np.loadtxt(f2,dtype=float, delimiter=',',skiprows=0)
testData = np.array(testData)
f2.close()

testRows = len(testData)

print("filas test",testRows)


# FIXME FALLA AL METER EN ESTE METODO LOS VALORES
# Funcion que multiplica los pesos por las entradas y devuelve una salida real 
def calcSalida(pesos, filadatos, umbral):
    # Realizamos la multiplicacion entre vectores (cada entrada con su peso correspondiente)
    matrizres = pesos*filadatos[:8]
    
    # Realizamos el sumatorio de la matriz resultante al multiplicar los pesos con las entradas para obtener la salida
    salidaReal = np.sum(matrizres, axis=1)

    # Convertimos a numero
    salidaReal = salidaReal[0]+umbral 

    return salidaReal

# /////////////////////////////////


matrizsalidas = np.empty(testRows)

for i in range(testRows):
     matrizsalidas[i] = calcSalida(pesos,testData[i], umbral)


print(matrizsalidas)

