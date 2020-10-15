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

#print("PESOS INICIALES: ",pesos)

# Funcion que multiplica los pesos por las entradas y devuelve una salida real 
def calcSalida(matrizpesos, filadatos, umbral):
    # Realizamos la multiplicacion entre vectores (cada entrada con su peso correspondiente)
    matrizres = matrizpesos*filadatos[:8]
    
    # Realizamos el sumatorio de la matriz resultante al multiplicar los pesos con las entradas para obtener la salida
    salidaReal = np.sum(matrizres, axis=1)

    salidaReal = salidaReal[0]+umbral # Convertimos a numero
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



# FIXME ARREGLAR METODO 
# Funcion Error Cuadratico medio MSE 
def calcMSE(columnadatos,matrizsalidas,numerofilas):
    print("COLUMNADATOS: ",columnadatos)
    print("MATRIZSALIDAS: ",matrizsalidas)

    np.array(columnadatos)
    np.array(matrizsalidas)
  
    matrizdiff = np.subtract(columnadatos,matrizsalidas)


    matrizdiff = columnadatos-matrizSalidas

    print("MATRIZdiff: ",matrizdiff)
    matrizdiff = matrizdiff**2
    resultadoN = np.sum(matrizdiff)
    print("RESULTADO N ",resultadoN)
    resultadoN = resultadoN/numerofilas
    print("Resultado del ERROR MSE ", resultadoN)
    return resultadoN





#######################################################  MAIN   ############################################# 
resultado = 0


""" PRUEBA DE CODIGO AÑADIR FILAS A UNA MATRIZ NADA QUE VER CON ESTE CODIGO
matrix = np.array([[1,2,3], [4,5,6]])

print("MATRIX NORMAL: ",matrix)

newrow = np.array([[1,2,3]])
matrix = np.concatenate((matrix,newrow))

print("MATRIX NUEVA: ",matrix)

"""

# TODO CREAR VARIABLE PARA EL NUMERO DE FILAS DE ENTRENAMIENTO, OTRA PARA TEST... (SUSTITUIR EL 10200)

# Aqui guardamos cada resultado con el objetivo de tener todas las salidas para el calculo de error
matrizSalidas = [] 
np.array(matrizSalidas)
""" MATRICES PARA GUARDAR TODOS LOS ERRORES DE CADA ITERACION Y POSTERIORMENTE HACER GRAFICAS...
matrizErroresAbsolutos = [nciclos]
matrizErroresCuadraticos = [nciclos]
"""

# Bucle for de los CICLOS
for i in range(nciclos):
    
    for j in range(10200):
        
        #print("resultado: ",resultado,", numero: ",j)
        matrizSalidas.append([])
        matrizSalidas[j].append(calcSalida(pesos,trainData[j],umbral))
        print(matrizSalidas[j])


resMSE = calcMSE(trainData[8],matrizSalidas,10200)

print("Resultados de MSE: ",resMSE)

        #pesos = calcNuevosPesos(pesos, trainData[j], razon, trainData[j][8], resultado)
        #umbral = calcNuevoUmbral(razon,resultado,trainData[j][8],umbral) 





#resultado = calcSalida(pesos,trainData[0],umbral)
#nuevosPesos = calcNuevosPesos(pesos, trainData[0], razon, trainData[0][8], resultado)
print(trainData[0])

#print("resultado y FINAL: ",resultado)
#print("pesos FINALES CALCULADOS: ",pesos)
#print("UMBRAL FINAL: ",umbral)




# FIXME PRUEBA DEL METODO MSE

matrizA = np.array([[1],[2],[3],[4]])
matrizB = np.array([[5],[1],[7],[3]])

resab = calcMSE(matrizA,matrizB,4)
print("TRESTTT ",resab)



# TODO VER QUE TIPO DE ERROR HAY QUE USAR EN ENTRENAMIENTO Y VALIDACION
# TODO ACABAR FORMULAS DE LOS ERRORES
# TODO CREAR MATRIZ DE SALIDAS POR CADA ITERACION PARA USARSE EN EL CALCULO DE ERRORES
# TODO VER COMO FUNCIONAN LOS CONJUNTOS DE TEST Y VALIDACION Y CUANDO HAY QUE USARLOS 
# TODO GRAFICAS CON LOS ERRORES DE TODAS LAS ITERACIONES