import streamlit as st
import cv2
import av
import urllib.request
from pathlib import Path
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import threading

# Configuración de la página
st.set_page_config(
    page_title="Detección facial",
    page_icon="📷",
    layout="wide"
)

st.title("Detección facial en tiempo real")
st.caption("Python · OpenCV · YuNet")

# Cargar modelo preentrenado
@st.cache_resource
def modelo():

    ruta = Path("face_detection_yunet_2023mar.onnx")

    if not ruta.exists():

        url = (
            "https://media.githubusercontent.com/media/"
            "opencv/opencv_zoo/main/models/"
            "face_detection_yunet/"
            "face_detection_yunet_2023mar.onnx"
        )

        urllib.request.urlretrieve(url, ruta)

    detector = cv2.FaceDetectorYN.create(
        str(ruta),
        "",
        (320, 320),
        0.6,
        0.3,
        5000
    )

    return detector, threading.Lock()


detector, lock = modelo()

# Procesamiento de video
def procesar(frame):

    img = frame.to_ndarray(format="bgr24")

    h, w = img.shape[:2]

    with lock:
        detector.setInputSize((w, h))
        _, rostros = detector.detect(img)

    total = 0 if rostros is None else len(rostros)

    if rostros is not None:

        for rostro in rostros:

            x, y, a, b = map(int, rostro[:4])

            confianza = float(rostro[-1])

            # Azul: confianza alta
            # Rosado: confianza moderada
            color = (
                (255, 0, 0)
                if confianza >= 0.85
                else (180, 105, 255)
            )

            cv2.rectangle(
                img,
                (x, y),
                (x + a, y + b),
                color,
                2
            )

            cv2.putText(
                img,
                f"Rostro {confianza:.1%}",
                (max(0, x), max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2
            )

    cv2.putText(
        img,
        f"Rostros: {total}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    return av.VideoFrame.from_ndarray(
        img,
        format="bgr24"
    )


# Configuración de servidores STUN
RTC_CONFIGURATION = {
    "iceServers": [
        {
            "urls": [
                "stun:stun.l.google.com:19302",
                "stun:stun1.l.google.com:19302",
                "stun:stun2.l.google.com:19302"
            ]
        }
    ]
}

# Cámara en tiempo real
st.subheader("Cámara en vivo")

webrtc_streamer(
    key="deteccion-facial",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_frame_callback=procesar,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 640},
            "height": {"ideal": 480},
            "frameRate": {"ideal": 15}
        },
        "audio": False
    },
    async_processing=True
)

# Información del modelo
st.divider()

st.subheader("Interpretación de resultados")

st.markdown("""
- 🔵 Azul: confianza facial igual o superior al 85%.
- 🩷 Rosado: confianza facial entre 60% y 85%.
- 🟢 Contador de rostros detectados.
""")

st.info(
    "Modelo YuNet preentrenado. "
    "Detección facial mediante OpenCV."
)

st.write('Azul: confianza ≥ 85 %. Rosado: confianza entre 60 % y 85 %.')
webrtc_streamer(key='yunet', mode=WebRtcMode.SENDRECV, video_frame_callback=procesar, media_stream_constraints={'video':True,'audio':False}, rtc_configuration={'iceServers':[{'urls':['stun:stun.l.google.com:19302']}]}, async_processing=True)
st.caption('La confianza facial no equivale al Accuracy ni determina el género. No se guardan imágenes en la aplicación.')
