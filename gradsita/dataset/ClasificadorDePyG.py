import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image

# Rutas
ruta_perros = "dataset/perros"
ruta_gatos = "dataset/gatos"

# Preparar listas
imagenes = []
etiquetas = []  # 0 = perro, 1 = gato

# Cargar imágenes de perros
for archivo in os.listdir(ruta_perros):
    if archivo.endswith((".jpg", ".png", ".jpeg")):
        img = Image.open(os.path.join(ruta_perros, archivo)).convert('L')  # gris
        img = img.resize((64, 64))  # tamaño fijo
        img_array = np.array(img).flatten()  # convertir a vector
        imagenes.append(img_array)
        etiquetas.append(0)

# Cargar imágenes de gatos
for archivo in os.listdir(ruta_gatos):
    if archivo.endswith((".jpg", ".png", ".jpeg")):
        img = Image.open(os.path.join(ruta_gatos, archivo)).convert('L')
        img = img.resize((64, 64))
        img_array = np.array(img).flatten()
        imagenes.append(img_array)
        etiquetas.append(1)

# Convertir a arrays de numpy
X = np.array(imagenes)
y = np.array(etiquetas)

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar clasificador
modelo = LogisticRegression(max_iter=1000)
modelo.fit(X_train, y_train)

# Evaluar
y_pred = modelo.predict(X_test)
print("Precisión:", accuracy_score(y_test, y_pred))

# Probar con una imagen nueva
imagen_prueba = Image.open("dataset/Imagendaprueba/prueba3.jpg").convert('L').resize((64, 64))
img_array = np.array(imagen_prueba).flatten().reshape(1, -1)
prediccion = modelo.predict(img_array)

print("Predicción:", "Gato 😺" if prediccion[0] == 1 else "Perro 🐶")
