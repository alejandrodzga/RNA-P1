# Practica 1 de Redes de Neuronas artificiales. Programacion de Adaline. Preprocesado de datos

import numpy as np
import random as rnd

# Funcion que coge los datos del archivo y los pasa en forma de matriz en numpy
def datainput():
    f = open('Data/california_housing_train.dat')
    data = np.loadtxt(f,dtype=float, delimiter=',',skiprows=1)
    np.array(data)
    f.close()
    return data 


# Funcion cuyo objetivo es normalizar los valores de la matriz entre 0 y 1
def normalizacion(data):
    #columna1
    v0 = data[:,0]
    maximo1 = np.amax(v0)
    minimo1 = np.amin(v0)
    data[:,0] = (v0-minimo1)/(maximo1-minimo1)

    #columna2
    v1 = data[:,1]
    maximo2 = np.amax(v1)
    minimo2 = np.amin(v1)
    data[:,1] = (v1-minimo2)/(maximo2-minimo2)

    #columna3
    v2 = data[:,2]
    maximo3 = np.amax(v2)
    minimo3 = np.amin(v2)
    data[:,2] = (v2-minimo3)/(maximo3-minimo3)

    #columna4
    v3 = data[:,3]
    maximo4 = np.amax(v3)
    minimo4 = np.amin(v3)
    data[:,3] = (v3-minimo4)/(maximo4-minimo4)

    #columna5
    v4 = data[:,4]
    maximo5 = np.amax(v4)
    minimo5 = np.amin(v4)
    data[:,4] = (v4-minimo5)/(maximo5-minimo5)

    #columna6
    v5 = data[:,5]
    maximo6 = np.amax(v5)
    minimo6 = np.amin(v5)
    data[:,5] = (v5-minimo6)/(maximo6-minimo6)

    #columna7
    v6 = data[:,6]
    maximo7 = np.amax(v6)
    minimo7 = np.amin(v6)
    data[:,6] = (v6-minimo7)/(maximo7-minimo7)

    #columna8
    v7 = data[:,7]
    maximo8 = np.amax(v7)
    minimo8 = np.amin(v7)
    data[:,7] = (v7-minimo8)/(maximo8-minimo8)

    #columna9
    v8 = data[:,8]
    maximo9 = np.amax(v8)
    minimo9 = np.amin(v8)
    data[:,8] = (v8-minimo9)/(maximo9-minimo9)




########################################################################################################
# CODIGO DE PREPROCESADO DE DATOS 

# Obtenemos la matriz a partir de la entrada de datos del fichero
datos = datainput()

# Normalizamos la matriz de datos
normalizacion(datos)

# Aleatorizacion de los datos (rotar las filas de la matriz)
np.random.shuffle(datos)

# Dividimos el conjunto de datos en 3 subconjuntos 
# Subconjunto de datos de entrenamiento
trainData = datos[:10200,]

# Subconjunto de datos de validacion
validationData = datos[10200:13600]

# Subconjunto de datos de test
testData = datos[13600:]


####################################### SALIDA DE DATOS #################################################


# SALIDA DEL CONJUNTO DE DATOS NORMALIZADOS (SIN DIVIDIR)
# Abrimos el archivo de salida de datos 
f1 = open("dataout.txt", "w")
# Guardamos la matriz en su formato en el archivo de salida de texto
np.savetxt(f1, datos, delimiter=' , ', fmt='%f')
f1.close()


# SALIDA DATOS DE ENTRENAMIENTO
# Abrimos el archivo de salida de datos 
f2 = open("trainData.txt", "w")
# Guardamos la matriz en su formato en el archivo de salida de texto
np.savetxt(f2, trainData, delimiter=' , ', fmt='%f')
f2.close()


# SALIDA DATOS DE VALIDACION
# Abrimos el archivo de salida de datos 
f3 = open("validationData.txt", "w")
# Guardamos la matriz en su formato en el archivo de salida de texto
np.savetxt(f3, validationData, delimiter=' , ', fmt='%f')
f3.close()


# SALIDA DATOS DE TEST
# Abrimos el archivo de salida de datos 
f4 = open("testData.txt", "w")
# Guardamos la matriz en su formato en el archivo de salida de texto
np.savetxt(f4, testData, delimiter=' , ', fmt='%f')
f4.close()





