
import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path

# ==========================================
# CONFIGURACION DE LA PAGINA
# ==========================================

st.set_page_config(
    page_title="Sistema de deteccion facial",
    page_icon="📷",
    layout="wide"
)

st.title("Sistema de detección facial")
st.caption("Inteligencia Artificial | OpenCV | MobileNetV2")

BASE = Path(__file__).resolve().parent

# ==========================================
# ESTILOS
# ==========================================

st.markdown("""
<style>
.stMetric {
    background-color: rgba(128,128,128,0.08);
    padding: 20px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# CARGAR METRICAS
# ==========================================

@st.cache_data
def cargar_metricas():
    with open(BASE / "metricas.json", encoding="utf-8") as archivo:
        return json.load(archivo)

try:
    metricas = cargar_metricas()

except Exception as error:
    st.error(f"Error al cargar las métricas: {error}")
    st.stop()

# ==========================================
# CARGAR MODELO
# ==========================================

@st.cache_resource
def cargar_modelo():
    return tf.keras.models.load_model(
        BASE / "modelo_men_women.keras",
        compile=False
    )

# ==========================================
# INDICADORES PRINCIPALES
# ==========================================

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Accuracy",
    f"{metricas['accuracy'] * 100:.2f}%"
)

col2.metric(
    "Imágenes evaluadas",
    metricas["total_imagenes"]
)

col3.metric(
    "Aciertos",
    metricas["aciertos"]
)

col4.metric(
    "Errores",
    metricas["errores"]
)

st.divider()

# ==========================================
# PESTAÑAS
# ==========================================

tab1, tab2, tab3 = st.tabs([
    "📷 Cámara en vivo",
    "📊 Evaluación del modelo",
    "🧠 Información del entrenamiento"
])

# ==========================================
# CAMARA EN TIEMPO REAL
# ==========================================

with tab1:

    st.subheader("Detección facial en tiempo real")

    st.caption(
        "Procesamiento de cámara mediante JavaScript "
        "y MediaPipe, sin WebRTC ni TURN."
    )

    ruta_camara = BASE / "camara.html"

    if ruta_camara.exists():

        html = ruta_camara.read_text(
            encoding="utf-8"
        )

        components.html(
            html,
            height=850,
            scrolling=True
        )

    else:
        st.error("No se encontró camara.html")

# ==========================================
# EVALUACION DEL MODELO
# ==========================================

with tab2:

    st.subheader("Resultados de clasificación")

    clases = metricas["clases"]

    matriz = np.array(
        metricas["matriz_confusion"]
    )

    izquierda, derecha = st.columns(2)

    with izquierda:

        st.markdown("### Matriz de confusión")

        fig, ax = plt.subplots(figsize=(5, 4))

        imagen = ax.imshow(
            matriz,
            cmap="Blues"
        )

        ax.set_xticks(range(len(clases)))
        ax.set_yticks(range(len(clases)))

        ax.set_xticklabels(clases)
        ax.set_yticklabels(clases)

        ax.set_xlabel("Predicción")
        ax.set_ylabel("Etiqueta del dataset")

        for i in range(len(clases)):
            for j in range(len(clases)):

                color = (
                    "white"
                    if matriz[i, j] > matriz.max() / 2
                    else "black"
                )

                ax.text(
                    j, i,
                    matriz[i, j],
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=15
                )

        fig.colorbar(imagen)
        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

    with derecha:

        st.markdown("### Aciertos y errores")

        fig2, ax2 = plt.subplots()

        ax2.pie(
            [
                metricas["aciertos"],
                metricas["errores"]
            ],
            labels=["Aciertos", "Errores"],
            colors=["#22c55e", "#ef4444"],
            autopct="%1.1f%%",
            startangle=90
        )

        st.pyplot(fig2)
        plt.close(fig2)

    st.divider()

    st.subheader("Reporte de clasificación")

    reporte = pd.DataFrame(
        metricas["reporte"]
    ).transpose()

    st.dataframe(
        reporte.round(4),
        use_container_width=True
    )

    st.subheader("Resultados por categoría")

    for clase in clases:

        datos = metricas["reporte"][clase]

        color = (
            "#2563eb"
            if clase.lower() == "man"
            else "#ec4899"
        )

        st.markdown(
            f"<h3 style='color:{color}'>{clase}</h3>",
            unsafe_allow_html=True
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Precision",
            f"{datos['precision'] * 100:.2f}%"
        )

        c2.metric(
            "Recall",
            f"{datos['recall'] * 100:.2f}%"
        )

        c3.metric(
            "F1-score",
            f"{datos['f1-score'] * 100:.2f}%"
        )

        st.progress(float(datos["recall"]))

    st.caption(
        "Las métricas representan la coincidencia "
        "con las etiquetas del conjunto de prueba."
    )

# ==========================================
# INFORMACION DEL ENTRENAMIENTO
# ==========================================

with tab3:

    st.subheader("Arquitectura MobileNetV2")

    informacion = {
        "Característica": [
            "Modelo",
            "Dataset",
            "Entrenamiento",
            "Validación",
            "Prueba",
            "Épocas",
            "Resolución",
            "Batch size",
            "Optimizador",
            "Accuracy"
        ],
        "Valor": [
            "MobileNetV2",
            "men-women.zip",
            "800 imágenes",
            "400 imágenes",
            "400 imágenes",
            "4",
            "224 x 224 x 3",
            "32",
            "Adam",
            f"{metricas['accuracy'] * 100:.2f}%"
        ]
    }

    st.dataframe(
        pd.DataFrame(informacion),
        hide_index=True,
        use_container_width=True
    )

    if st.button("Verificar modelo entrenado"):

        try:
            modelo = cargar_modelo()

            st.success("Modelo cargado correctamente")

            st.write("Entrada:", modelo.input_shape)
            st.write("Salida:", modelo.output_shape)
            st.write(
                "Parámetros totales:",
                modelo.count_params()
            )

        except Exception as error:
            st.error(str(error))

    ruta_modelo = BASE / "modelo_men_women.keras"

    if ruta_modelo.exists():

        st.download_button(
            "Descargar modelo entrenado",
            data=ruta_modelo.read_bytes(),
            file_name="modelo_men_women.keras"
        )

st.divider()

st.caption(
    "Proyecto académico de Inteligencia Artificial"
)
