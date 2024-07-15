import os
import random


# Balance the error file so that the number of animal images is equal to the number of empty images
def main():
    

    animalProportion = 24  # TODO: Change depending on the dataset and the number of empty and animal images


    blockWidth = 6
    blockHeight = 4

    numDivisiones = blockHeight * blockWidth

    errorsFile = "./Data/TrainMulticlase.txt"
    balancedErrorsFile = "./Data/TrainMulticlase_Balanced.txt"

    

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

        main()