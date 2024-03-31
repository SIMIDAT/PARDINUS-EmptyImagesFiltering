import os
import random

import config


# Balanceado teniendo en cuenta que la proporcion de animales es el 22%. Nos quedamos con el 22% de las imágenes vacías.
def main():

    print("Balancear...")
    
    trainTest = config.TRAINTEST
    numberOfClusters = config.NUMBER_OF_CLUSTERS

    animalProportion = config.ANIMAL_PROPORTION


    blockWidth = config.BLOCK_WIDTH
    blockHeight = config.BLOCK_HEIGHT

    numDivisiones = blockHeight * blockWidth

    errorsFile = "./ErrorFiles/" + trainTest + "_Errors_" + str(numberOfClusters) + "_RAE.txt"
    balancedErrorsFile = "./ErrorFiles/" + trainTest + "_Errors_" + str(numberOfClusters) + "_RAE_balanced.txt"

    

    fichero = open(errorsFile, "r")
    ficheroBalanceado = open(balancedErrorsFile, "w")

    lineasErrores = fichero.readlines()

    contadorVacio = 0
    contadorAnimales = 0

    for lineaFichero in lineasErrores:
        linea = lineaFichero.rstrip()

        linea = linea.split(",")

        if linea[-1] == '':
            linea.pop()

        # If it is a empty image, throw random
        if linea[numDivisiones * 3 + 1] == "0":
            aleatorio = random.randint(0, 100)

            if aleatorio < animalProportion:
                ficheroBalanceado.write(lineaFichero)
                contadorVacio += 1

        # If it is an animal image, we add it
        elif linea[numDivisiones * 3 + 1] == "1":
            ficheroBalanceado.write(lineaFichero)
            contadorAnimales += 1

        else:
            print("You should not see this DDD: (balanceErrorFile.py)")
            print(linea)

    print("Empty: " + str(contadorVacio) + " -- Animal: " + str(contadorAnimales))

    ficheroBalanceado.close()
    fichero.close()



if __name__ == "__main__":

    if not config.BALANCE:
        print("ERROR: You have to set 'config.py > BALANCE=True'")
    else:
        main()