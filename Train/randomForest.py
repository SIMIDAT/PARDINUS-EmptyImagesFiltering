from sklearn.ensemble import RandomForestClassifier
import pickle as pkl
import random



# Prepare data for traininig

listaErroresEtiquetas = list()

# Read balanced file
balancedTrainErrorFile = "./Data/TrainMulticlase_Balanceado.txt"

f = open(balancedTrainErrorFile, "r")

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


# Train the model

randomForestModel = RandomForestClassifier(n_estimators=200)
randomForestModel.fit(listaErrores, listaEtiquetas)

# Save the model
pkl.dump(randomForestModel, open("./modeloRF.pkl", "wb"))



