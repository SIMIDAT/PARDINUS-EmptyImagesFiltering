from sklearn.ensemble import RandomForestClassifier

import config
import random





# Prepare data for traininig

listaErroresEtiquetas = list()

# Read empty file
emptyTrainErrorFile = config.EMPTY_TRAIN_ERROR_FILE

# Read animal file
animalTrainErrorFile = config.ANIMAL_TRAIN_ERROR_FILE

f = open(animalTrainErrorFile, "r")

lineas = f.readlines()

c = 0
for linea in lineas:
    linea = linea.rstrip()
    listaErroresEtiquetas.append(linea)

    c+=1

f.close()
print(c)


# Shuffle elements

random.shuffle(listaErroresEtiquetas)

# Separate data and labels

listaErrores = list()
listaEtiquetas = list()

for error in listaErroresEtiquetas:
    errorSplit = error.split(",") 
    errorSplit.pop()  # Pop to remove an empty data

    # Get label
    etiqueta = str(errorSplit[-1])
    listaEtiquetas.append(etiqueta)
    errorSplit.pop()

    # Get errors
    errorLineNumber = [float(i) for i in errorSplit]
    errorLineNumber[-1] = int(errorLineNumber[-1])
    listaErrores.append(errorLineNumber)

print(listaErrores)
print(listaEtiquetas)

# Train the model

#randomForestModel = RandomForestClassifier(n_estimators=200)

# Save the model
