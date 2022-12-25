import streamlit as st
import cv2
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import io

image_list = []
image_name = []

dict_map = {
                "0": "safe driving",
                "1": "texting - right",
                "2": "talking on the phone - right",
                "3": "texting - left",
                "4": "talking on the phone - left",
                "5": "operating the radio",
                "6": "drinking",
                "7": "reaching behind",
                "8": "hair and makeup",
                "9": "talking to passenger"
            }

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('best_model.h5')
    return model

st.write("""

         # Prediction Distracted Driver

         """)
st.write("State Farm Distracted Driver Detection Can computer vision spot distracted drivers?")
file = st.file_uploader("Please upload an image file", type=["jpg", "png"], accept_multiple_files=True)


def import_and_predict(image_data, model):    
    size = (400,400)    
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resize = (cv2.resize(img, dsize=(400, 400)))
        
    img_reshape = img_resize[np.newaxis]
    
    prediction = model.predict(img_reshape)
        
    return prediction


model = load_model()
for upload in file:        
    bytes_data = upload.read()
    image = Image.open(io.BytesIO(bytes_data))
    st.image(image, use_column_width=False, width=400)
    st.write("filename:", upload.name)
    image_name.append(upload.name)
    predictions = import_and_predict(image, model)
    for item in predictions.argmax(axis=1):
        image_list.append(dict_map[str(item)])

st.write(" ")
st.write(" ")
st.write(" ")

with st.container():
    for i in range(len(image_list)):
        st.write("{}  -->  {}".format(image_name[i], image_list[i]))
