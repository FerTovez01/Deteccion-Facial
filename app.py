import streamlit as st
import cv2
import av
import urllib.request
from pathlib import Path
import threading
import numpy as np
from streamlit_webrtc import webrtc_streamer, WebRtcMode

st.set_page_config(page_title='Detección facial', page_icon='📷', layout='wide')
st.title('Detección y clasificación facial')

@st.cache_resource
def cargar_detector():
    ruta = Path('yunet.onnx')
    if not ruta.exists():
        urllib.request.urlretrieve('https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx', ruta)
    return cv2.FaceDetectorYN.create(str(ruta), '', (320,320), 0.6, 0.3, 5000), threading.Lock()

def procesar(frame):
    img = frame.to_ndarray(format='bgr24')
    detector, lock = cargar_detector()
    h,w = img.shape[:2]
    with lock:
        detector.setInputSize((w,h))
        _, rostros = detector.detect(img)
    for f in ([] if rostros is None else rostros):
        x,y,a,b = map(int,f[:4])
        cv2.rectangle(img,(x,y),(x+a,y+b),(0,255,0),2)
        cv2.putText(img,f'Rostro {f[-1]:.1%}',(max(0,x),max(20,y-8)),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,0),2)
    return av.VideoFrame.from_ndarray(img,format='bgr24')

@st.cache_resource
def cargar_modelo():
    import tensorflow as tf
    return tf.keras.models.load_model('modelo_genero.keras',compile=False)

@st.cache_data
def prueba():
    from datasets import load_dataset
    from sklearn.model_selection import train_test_split
    ds = load_dataset('HuggingFaceM4/FairFace','0.25',split='validation',streaming=True).shuffle(seed=42,buffer_size=1000)
    X,y = [],[]
    for fila in ds:
        if fila['age'] < 3: continue
        X.append(np.asarray(fila['image'].convert('RGB').resize((224,224))))
        y.append(fila['gender'])
        if len(y)==800: break
    _, X_test, _, y_test = train_test_split(np.array(X),np.array(y),test_size=.5,random_state=42,stratify=y)
    return X_test,y_test

camara, evaluacion = st.tabs(['Cámara en vivo','Evaluación FairFace'])
with camara:
    st.caption('YuNet: detección facial, sin clasificación de género en cámara.')
    cargar_detector()
    webrtc_streamer(key='camara',mode=WebRtcMode.SENDRECV,rtc_configuration={'iceServers':[{'urls':['stun:stun.l.google.com:19302','stun:stun1.l.google.com:19302']}]},video_frame_callback=procesar,media_stream_constraints={'video':True,'audio':False},async_processing=True)
with evaluacion:
    st.caption('Predicciones de etiquetas del dataset, no determinación de identidad de género.')
    if st.button('Evaluar modelo'):
        try:
            from sklearn.metrics import confusion_matrix, accuracy_score
            with st.spinner('Cargando conjunto de prueba y evaluando...'):
                X,y = prueba()
                p = cargar_modelo().predict(X,batch_size=32,verbose=0)
                pred = p.argmax(axis=1)
            st.session_state['resultado'] = (y,p,pred)
        except Exception as e:
            st.error(str(e))
    if 'resultado' in st.session_state:
        from sklearn.metrics import confusion_matrix, accuracy_score
        y,p,pred = st.session_state['resultado']
        st.metric('Accuracy de prueba',f'{accuracy_score(y,pred):.2%}')
        st.write('Matriz de confusión (filas reales, columnas predichas; orden Male, Female)')
        st.dataframe(confusion_matrix(y,pred,labels=[0,1]))
        i = st.slider('Muestra de prueba',0,len(y)-1,0)
        etiqueta = ['Male','Female'][pred[i]]
        color = '#2563eb' if pred[i]==0 else '#db2777'
        st.markdown(f'<div style="border:3px solid {color};padding:12px;color:{color}">{etiqueta}: {p[i,pred[i]]:.1%}</div>',unsafe_allow_html=True)
        st.write(f'Male: {p[i,0]:.1%} | Female: {p[i,1]:.1%}')
        st.write('Etiqueta original:', ['Male','Female'][y[i]])
