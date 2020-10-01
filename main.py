# Practica 1 de Redes de Neuronas artificiales
import numpy as np


# Funcion que coge los datos del archivo y los pasa en forma de matriz en numpy
def datainput():
    f = open('Data/california_housing_train.dat')
    data = np.loadtxt(f,dtype=float, delimiter=',',skiprows=1)
    f.closed
    return data 

# Funcion cuyo objetivo es normalizar los valores de la matriz entre 0 y 1
def normalizacion(data):
    #columna1
    maximo1 = np.amax(data[:,0])
    minimo1 = np.amin(data[:,0])
    print("hola: ",maximo1)
    print("hola2: ",minimo1)
    data[:,0] = (data[:,0]-minimo1)/(maximo1-minimo1)
    #columna2
    maximo2 = np.amax(data[:,1])
    minimo2 = np.amin(data[:,1])
    print("hola3: ",maximo2)
    print("hola4: ",minimo2)
    data[:,1] = (data[:,1]-minimo2)/(maximo2-minimo2)

    #columna3
    maximo3 = np.amax(data[:,2])
    minimo3 = np.amin(data[:,2])
    #columna4
    maximo4 = np.amax(data[:,3])
    minimo4 = np.amin(data[:,3])
    #columna5
    maximo5 = np.amax(data[:,4])
    minimo5 = np.amin(data[:,4])
    #columna6
    maximo6 = np.amax(data[:,5])
    minimo6 = np.amin(data[:,5])
    #columna7
    maximo7 = np.amax(data[:,6])
    minimo7 = np.amin(data[:,6])
    #columna8
    maximo8 = np.amax(data[:,7])
    minimo8 = np.amin(data[:,7])


# Funcion que define la formula para la normalizacion de los valores de la matriz 
#def normaliza_valor(varoriginal,varmax,varmin):



#######################################################
print('METODO MAIN AQUI:')

datos = datainput()

normalizacion(datos)

print(np.array(datos[:,1]))

