import os
import random

import config

# Balance the error file so that the number of animal images is equal to the number of empty images
def main():
    

    animalProportion = config.ANIMAL_PROPORTION  # Change depending on the dataset and the number of empty and animal images

    numberOfClusters = config.NUMBER_OF_CLUSTERS

    emptyFile = config.ERROR_FILES_ROUTE + "Train_Error_" + str(numberOfClusters) + "_Empty.txt"
    animalFile = config.ERROR_FILES_ROUTE + "Train_Error_" + str(numberOfClusters) + "_Animal.txt"
    balancedErrorsFile = config.ERROR_FILES_ROUTE + "Train_Error_" + str(numberOfClusters) + "_Balanced.txt"



    ficheroVacio = open(emptyFile, "r")
    ficheroAnimales = open(animalFile, "r")
    ficheroBalanceado = open(balancedErrorsFile, "w")


    lineasErroresVacio = ficheroVacio.readlines()
    lineasErroresAnimales = ficheroAnimales.readlines()


    contadorVacio = 0
    contadorAnimales = 1

    # If it is an animal image, we add it
    for lineaFichero in lineasErroresAnimales:
        linea = lineaFichero.rstrip()
        linea = linea.split(",")
        if linea[-1] == '':
            linea.pop()

        ficheroBalanceado.write(lineaFichero)
        contadorAnimales += 1

    # If it is an empty image, throw random
    for lineaFichero in lineasErroresVacio:
        linea = lineaFichero.rstrip()
        linea = linea.split(",")
        if linea[-1] == '':
            linea.pop()

        aleatorio = random.randint(0, 100)

        if aleatorio < animalProportion:
            ficheroBalanceado.write(lineaFichero)
            contadorVacio += 1


    ficheroVacio.close()
    ficheroAnimales.close()
    ficheroBalanceado.close()



if __name__ == "__main__":

        main()