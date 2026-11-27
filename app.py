import streamlit as st
from pathlib import Path
import numpy as np
import streamlit.components.v1 as components

st.set_page_config(page_title='Detección facial', page_icon='📷', layout='wide')
st.title('Detección y clasificación facial')

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
    st.caption('Cámara procesada localmente en el navegador mediante MediaPipe. No utiliza WebRTC ni TURN.')
    components.html(Path(__file__).with_name('camara.html').read_text(encoding='utf-8'), height=650)
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
