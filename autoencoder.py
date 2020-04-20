"""
Deteccion de fraudes en transacciones con tarjetas de credito
utilizando un autoencoder
"""

# 1. LECTURA DEL DATASET
import pandas as pd
import matplotlib.pyplot as plt

datos = pd.read_csv("creditcard.csv")
print(datos.head())

# Determinamos el numero de clases (target)
# y la cantidad de transacciones por clase
# 1 - transacciones fraudulentas
# 0 - transacciones normales
nr_clases = datos['Class'].value_counts(sort=True)
print(nr_clases)

# 2. ANALISIS EXPLORATORIO DE LOS DATOS

# Cantidad de registros normales vs fraudulentos
nr_clases.plot(kind = 'bar', rot = 0)
plt.xticks(range(2),['Normales','Fraudulentos'])
plt.title("Distribucion de los datos")
plt.xlabel("Clase")
plt.ylabel("Cantidad")
plt.show()

# Monto de transacciones vs tiempo
normales = datos[datos.Class == 0]
fraudulentos = datos[datos.Class == 1]
plt.scatter(normales.Time/3600, normales.Amount, alpha = 0.5,
            c = '#19323C', label = 'Normales', s = 3)
plt.scatter(fraudulentos.Time/3600, fraudulentos.Amount, alpha = 0.5,
            c = '#F2545B', label = 'Fraudulentos', s = 3)
plt.xlabel('Tiempo desde la primera transaccion (h)')
plt.ylabel('Monto (Euros)')
plt.legend(loc = 'upper right')
plt.show()

# Distribucion de caracteristicas V1 a V28 en normales y fraudulentos
import matplotlib.gridspec as gridspec
import seaborn as sns

v_1_28 = datos.iloc[:,1:29].columns
gs = gridspec.GridSpec(28,1)
for i, cn in enumerate(datos[v_1_28]):
    sns.distplot(datos[cn][datos.Class == 1], bins = 50,
                 label = 'Fraudulentos', color = '#F2545B')
    sns.distplot(datos[cn][datos.Class == 0], bins = 50,
                 label = 'Normales', color = '#19323C')
    plt.xlabel('')
    plt.title('Histograma caracteristica: ' + str(cn))
    plt.legend(loc = 'upper right')
    plt.show()

# 3. PREPROCESAMIENTO DE LOS DATOS

#3.1 - La variable "Tiempo" al no aportar informacion que diferencie las
# distribuciones, se elimina
datos.drop(['Time'], axis = 1, inplace = True)

# Normalizamos la variable de montos
# Recomendacion: Normalizar todas las variables para tener una misma unidad
from sklearn.preprocessing import StandardScaler
datos['Amount'] = StandardScaler().fit_transform(datos['Amount'].values.reshape(-1,1))

# Particionamos los datos de entrenamiento y prueba
from sklearn.model_selection import train_test_split

X_train, X_test = train_test_split(datos, test_size=0.2, random_state=42)

# Selecciono unicamente las transacciones normales (0) para entrenar al modelo
# La idea es que genere un error de prediccion alto al intentar predecir registros
# fraudulentos (1)
X_train = X_train[X_train.Class == 0]
X_train = X_train.drop(['Class'], axis = 1)     # Elimino la variable objetivo
X_train = X_train.values

# Defino la variable objetivo con los datos de prueba
Y_test = X_test['Class']

X_test = X_test.drop(['Class'], axis = 1)   # Elimino la variable objetivo
X_test = X_test.values

# 4. CONSTRUCCION DEL AUTOENCODER 29-20-14-20-29, TANH-RELU-TANH-RELU
import numpy as np
np.random.seed(5)

#import keras
#import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, Dense

#config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 6} )
#sess = tf.compat.v1.Session(config=config)
#tf.compat.v1.keras.backend.get_session(sess)

dim_entrada = X_train.shape[1]  # 29 categorias quitando la variable tiempo y el target
capa_entrada = Input(shape=(dim_entrada,))

encoder = Dense(20, activation='tanh')(capa_entrada)
encoder = Dense(14, activation='relu')(encoder)

decoder = Dense(20, activation='tanh')(encoder)
decoder = Dense(29, activation = 'relu')(encoder)

autoencoder = Model(inputs = capa_entrada, outputs = decoder)

from keras.optimizers import SGD
sgd = SGD(lr=0.01)
autoencoder.compile(optimizer = 'sgd', loss = 'mse')

nits = 100
tam_lote = 32
autoencoder.fit(X_train, X_train, epochs = nits, batch_size=tam_lote, shuffle=True,
                validation_data=(X_test, X_test), verbose = 1)

# 5. VALIDACION
# Prediccion con los datos de prueba
X_pred = autoencoder.predict(X_test)
# Calculo de Error Cuadratico Medio
ecm = np.mean(np.power(X_test-X_pred,2), axis = 1)
print(X_pred.shape)

# Grafica precision y recall para determinar el umbral que detecte transacciones fraudulentas

from sklearn.metrics import confusion_matrix, precision_recall_curve
precision, recall, umbral = precision_recall_curve(Y_test, ecm)

plt.plot(umbral, precision[1:], label = "Precision", linewidth = 5)
plt.plot(umbral, recall[1:], label = "Recall", linewidth = 5)
plt.title('Precision y Recall para diferentes umbrales')
plt.xlabel('Umbral')
plt.ylabel('Precision/Recall')
plt.legend()
plt.show()

# Implementamos la matriz de confusion
umbral_fijo = 0.75
Y_pred = [1 if e > umbral_fijo else 0 for e in ecm]

conf_matrix = confusion_matrix(Y_test, Y_pred)
print(conf_matrix)




