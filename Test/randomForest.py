import config
import utils

import os
import pickle
import shutil
import cv2 as cv

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix







# Loading KMeans model (sklearn 1.2.0)
rfModel = pickle.load(open(config.RF_ROUTE, "rb"))


# Load test data
# TODO: dejar esta línea --> f = open(config.ERRORS_DIRECTORY + os.sep + "Test_Errors_7_RAE_" + config.DATA_NAME + ".txt")
f = open(config.ERRORS_DIRECTORY + os.sep + "Tercera_Test_SinEtiquetar.txt")

errorList = f.readlines()

f.close()

listaEtiquetasPredichas = list()

listaErroresAPredecir = list()
listaNombres = list()

for errorLine in errorList:
    errorLine = errorLine.rstrip()

    errorLine = errorLine.split(",")

    # Obtenemos nombre del archivo y lo eliminamos de la lista de errores
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

print(etiquetasPrediccion)


# GUARDAMOS LOS RESULTADOS
f = open("Resultados_Animales.txt", "w")

for i, etiqueta in enumerate(etiquetasPrediccion):
    f.write(str(etiqueta) + "," + listaNombres[i])

f.close()

# EVALUATION
# TODO: Borrar

etiquetasTest = list()

for i in range(3000):
    etiquetasTest.append(0)

for i in range(3000):
    etiquetasTest.append(1)

print(len(etiquetasTest))

print(classification_report(etiquetasTest, etiquetasPrediccion))

#confusion_matrix(etiquetasTest, etiquetasPrediccion)

a = confusion_matrix(etiquetasTest, etiquetasPrediccion)

tp = round((float(a[0][0]) / 3000) * 100, 4)
fp = round((float(a[0][1]) / 3000) * 100, 4)
fn = round((float(a[1][0]) / 3000) * 100, 4)
tn = round((float(a[1][1]) / 3000) * 100, 4)

print("Matriz de confusión")

print("TP  FP")
print("FN  TN\n")
print(str(tp) + "   " + str(fp))
print(str(fn) + " " + str(tn))

print()
print(fn, fp)
    
    