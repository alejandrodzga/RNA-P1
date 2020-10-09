# Practica 1 de Redes de Neuronas artificiales. Programacion de Adaline. 
import numpy as np
import sys
import random

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

#print("LOS PESOS: ",type(pesos[0,6]),pesos)
#print("UMBRAL: ",type(umbral))


# Caso de error en el que no se le pasan los parametros necesarios 
if(len(sys.argv)!=3):
    raise TypeError("Numero incorrecto de parametros!")


# Entrada de los hiperparametros 
# El formato de entrada de parametros es el siguiente: Adaline.py nciclos razon
nciclos = int(sys.argv[1])
razon = float(sys.argv[2])

print("PESOS INICIALES: ",pesos)

# Funcion que multiplica los pesos por las entradas y devuelve una salida real 
def calcSalida(matrizpesos, filadatos, umbral):
    # Realizamos la multiplicacion entre vectores (cada entrada con su peso correspondiente)
    matrizres = matrizpesos*filadatos[:8]
    
    # Realizamos el sumatorio de la matriz resultante al multiplicar los pesos con las entradas para obtener la salida
    salidaReal = np.sum(matrizres, axis=1)

    salidaReal = salidaReal[0]+umbral # Convertimos a numero
    return salidaReal


def calcNuevosPesos(matrizpesos, filadatos, razon, resultadoEsperado, resultado):
    #ERROR: Tipo de datos (float, long) incompatibles a la hora de operar
    print("filadatos: ",filadatos[:8])
    matrizIncrementoPesos = (razon*(resultadoEsperado-resultado))*filadatos[:8]
    print("MATRIZ INCREMENTO DE PESOS: ",matrizIncrementoPesos)
    matrizNuevosPesos = matrizpesos + matrizIncrementoPesos




    return matrizNuevosPesos

##def calcNuevoUmbral(razon, resultado, resultadoEsperado):

#######################################################  MAIN   ############################################# 


resultado = calcSalida(pesos,trainData[0],umbral)
nuevosPesos = calcNuevosPesos(pesos, trainData[0], razon, trainData[0][8], resultado)
print(trainData[0])

print("resultado y: ",resultado)
print("Nuevos pesos CALCULADOS: ",nuevosPesos)