import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import streamlit as st

st.title("Clasificador de Perros y Gatos 🐶🐱")
st.write("Este modelo usa regresión logística para clasificar imágenes.")

# Rutas
ruta_perros = "gradsita/dataset/perros"
ruta_gatos = "gradsita/dataset/gatos"

imagenes = []
etiquetas = []  # 0 = perro, 1 = gato

# Verificar si las carpetas existen
if not os.path.exists(ruta_perros) or not os.path.exists(ruta_gatos):
    st.error("❌ No se encontraron las carpetas de datos. Verifica las rutas.")
    st.stop()

# Cargar imágenes de perros
st.write("Cargando imágenes de perros...")
for archivo in os.listdir(ruta_perros):
    if archivo.endswith((".jpg", ".png", ".jpeg")):
        img = Image.open(os.path.join(ruta_perros, archivo)).convert('L')
        img = img.resize((64, 64))
        img_array = np.array(img).flatten()
        imagenes.append(img_array)
        etiquetas.append(0)

# Cargar imágenes de gatos
st.write("Cargando imágenes de gatos...")
for archivo in os.listdir(ruta_gatos):
    if archivo.endswith((".jpg", ".png", ".jpeg")):
        img = Image.open(os.path.join(ruta_gatos, archivo)).convert('L')
        img = img.resize((64, 64))
        img_array = np.array(img).flatten()
        imagenes.append(img_array)
        etiquetas.append(1)

# Entrenar modelo
st.write("Entrenando modelo...")
X = np.array(imagenes)
y = np.array(etiquetas)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = LogisticRegression(max_iter=1000)
modelo.fit(X_train, y_train)

# Evaluar
y_pred = modelo.predict(X_test)
precision = accuracy_score(y_test, y_pred)
st.success(f"✅ Precisión del modelo: {precision*100:.2f}%")

# Probar con una imagen nueva
st.subheader("Prueba con una imagen nueva 📸")
ruta_prueba = "gradsita/dataset/Imagendaprueba/prueba3.jpg"

if os.path.exists(ruta_prueba):
    imagen_prueba = Image.open(ruta_prueba).convert('L').resize((64, 64))
    st.image(imagen_prueba, caption="Imagen de prueba", use_column_width=True)
    img_array = np.array(imagen_prueba).flatten().reshape(1, -1)
    prediccion = modelo.predict(img_array)
    resultado = "😺 Gato" if prediccion[0] == 1 else "🐶 Perro"
    st.info(f"Predicción del modelo: {resultado}")
else:
    st.warning("⚠️ No se encontró la imagen de prueba.")
