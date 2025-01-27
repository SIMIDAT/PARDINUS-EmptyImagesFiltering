import config
import utils

import os
import pickle
import shutil
import cv2 as cv

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix




'''
Use a RandomForest model with reconstruction error instances to determine if an instance is empty or not.

'''




def main():

    # Loading KMeans model (sklearn 1.2.0)
    rfModel = pickle.load(open(config.TRAINED_MODELS_ROUTE + "./modeloRF.pkl", "rb"))


    # Load test data
    #TODO
    f = open(config.ERRORS_DIRECTORY + os.sep + "Test_Name_Errors_7_RAE.txt")

    errorList = f.readlines()

    f.close()

    listaEtiquetasPredichas = list()

    listaErroresAPredecir = list()
    listaNombres = list()

    for errorLine in errorList:
        errorLine = errorLine.rstrip()

        errorLine = errorLine.split(",")

        # Get filename
        nombreArchivo = errorLine[-1]
        errorLine.pop()
        listaNombres.append(nombreArchivo)

        #print(errorLine)
        errorLineNumber = [float(i) for i in errorLine]
        errorLineNumber[-1] = int(errorLineNumber[-1])
        #print(errorLineNumber)

        listaErroresAPredecir.append(errorLineNumber)

    listaEtiquetasPredichas = rfModel.predict_proba(listaErroresAPredecir)

    etiquetasPrediccion = list()

    for etiqueta in listaEtiquetasPredichas:
        if etiqueta[0] < etiqueta[1]:  # [0,1] Animales
            # Animal
            etiquetasPrediccion.append(1)
        else:                                                      # [1,0] Empty
            # Empty
            etiquetasPrediccion.append(0)

    # print(etiquetasPrediccion)


    # GUARDAMOS LOS RESULTADOS
    f = open("./Results.csv", "w")

    for i, etiqueta in enumerate(etiquetasPrediccion):
        f.write(listaNombres[i] + ";" + str(etiqueta) + "\n")

    f.close()



if __name__ == "__main__":

    main()

    
    