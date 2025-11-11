import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import streamlit as st

st.title("Clasificador de Perros y Gatos 🐶🐱")
st.write("Este modelo usa regresión logística para clasificar imágenes de perros y gatos.")

# === Rutas ===
ruta_perros = "gradsita/dataset/perros"
ruta_gatos = "gradsita/dataset/gatos"
ruta_pruebas = "gradsita/dataset/Imagendaprueba"

# === Verificar carpetas ===
for ruta in [ruta_perros, ruta_gatos, ruta_pruebas]:
    if not os.path.exists(ruta):
        st.error(f"❌ No se encontró la carpeta: {ruta}")
        st.stop()

# === Cargar imágenes de entrenamiento ===
imagenes = []
etiquetas = []

st.write("📦 Cargando imágenes...")

for archivo in os.listdir(ruta_perros):
    if archivo.lower().endswith((".jpg", ".jpeg", ".png")):
        img = Image.open(os.path.join(ruta_perros, archivo)).convert('L').resize((64, 64))
        imagenes.append(np.array(img).flatten())
        etiquetas.append(0)

for archivo in os.listdir(ruta_gatos):
    if archivo.lower().endswith((".jpg", ".jpeg", ".png")):
        img = Image.open(os.path.join(ruta_gatos, archivo)).convert('L').resize((64, 64))
        imagenes.append(np.array(img).flatten())
        etiquetas.append(1)

# === Entrenar modelo ===
st.write("🤖 Entrenando modelo...")
X = np.array(imagenes)
y = np.array(etiquetas)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = LogisticRegression(max_iter=1000)
modelo.fit(X_train, y_train)

precision = accuracy_score(y_test, modelo.predict(X_test))
st.success(f"✅ Precisión del modelo: {precision*100:.2f}%")

# === Seleccionar imagen de prueba ===
st.subheader("📸 Selecciona una imagen para probar")

imagenes_disponibles = [f for f in os.listdir(ruta_pruebas) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

if len(imagenes_disponibles) == 0:
    st.warning("⚠️ No hay imágenes en la carpeta de pruebas.")
else:
    imagen_seleccionada = st.selectbox("Elige una imagen:", imagenes_disponibles)

    if st.button("Clasificar imagen seleccionada"):
        ruta_imagen = os.path.join(ruta_pruebas, imagen_seleccionada)
        imagen_prueba = Image.open(ruta_imagen).convert('L').resize((64, 64))

        st.image(ruta_imagen, caption="Imagen seleccionada", use_column_width=True)

        img_array = np.array(imagen_prueba).flatten().reshape(1, -1)
        prediccion = modelo.predict(img_array)

        resultado = "😺 Es un gato" if prediccion[0] == 1 else "🐶 Es un perro"
        st.info(resultado)
