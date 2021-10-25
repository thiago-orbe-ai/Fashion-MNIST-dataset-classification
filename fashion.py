#Importar Bibliotecas e dependências

import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import io
from tensorflow import keras
import cnn_model
import Seq_model
import pandas as pd
import pickle
import time
import numpy as np
from PIL import Image, ImageOps

fas_data = keras.datasets.fashion_mnist #carregar dataset
(train_images, train_labels), (test_images, test_labels) = fas_data.load_data() #carregar dados de treino e teste
seq_model = tf.keras.models.load_model("Seq_model") #rede neural ann (sequencial)
cnn_model = tf.keras.models.load_model("cnn_model") #rede neural convolucional (conv, pooling, fully connected layers) 
class_names = ['Tshirt/TOP','Trouser','Pullover','Dress','Coat', 'Sandel','Shirt','Sneaker','Bag','Ankle boot'] #tipos de peças

#Configuração da Barra Lateral
add_selectbox = st.sidebar.selectbox('select the model for classification', ('Sequential', 'CNN'))

#Título do App
st.title("Classificação de pelas - Fashion MNIST dataset")

#Funções
def explore_data(train_images,train_label,test_images):
    st.write('Train Images shape:',train_images.shape)
    st.write('Test images shape:',test_images.shape)
    st.write('Training Classes',len(np.unique(train_labels)))
    st.write('Testing Classes',len(np.unique(test_labels)))

def  CNN_model_summary():
    img=Image.open("cnn_summary.PNG")
    st.image(img)

def  Seq_model_Summary():
    img=Image.open("Seq_summary.PNG")
    st.image(img)
    
def seq_history_graph():
    infile=open('seq_trainHistory',"rb")
    history = pickle.load(infile)
    plt.figure(figsize=(7,7))
    train_acc=history['accuracy']
    val_acc=history['val_accuracy']
    train_loss=history['loss']
    val_loss=history['val_loss']
    plt.subplot(2,1,1)
    plt.plot(train_acc,label='Training accuracy')
    plt.plot(val_acc,label='Validation accuracy')
    plt.legend()
    plt.title('acc')
    plt.subplot(2,1,2)
    plt.plot(train_loss,label='Training loss')
    plt.plot(val_loss,label='Validation loss')
    plt.legend()
    plt.title('loss')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()                
    
def cnn_history_graph():
    infile=open('cnntrainHistory',"rb")
    history = pickle.load(infile)
    plt.figure(figsize=(7,7))
    train_acc=history['accuracy']
    val_acc=history['val_accuracy']
    train_loss=history['loss']
    val_loss=history['val_loss']
    plt.subplot(2,1,1)
    plt.plot(train_acc,label='Training accuracy')
    plt.plot(val_acc,label='Validation accuracy')
    plt.legend()
    plt.title('acc')
    plt.subplot(2,1,2)
    plt.plot(train_loss,label='Training loss')
    plt.plot(val_loss,label='Validation loss')
    plt.legend()
    plt.title('loss')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
    
def cnn_archi():
    img=Image.open('cnn_model_architecture.png')
    st.image(img)
def seq_archi():
    img=Image.open('seq_model_architecture.png')
    st.image(img)

#Carregar arquivo

if(add_selectbox=='CNN' or add_selectbox=='Sequential'):
    file_uploader=st.file_uploader('Upload cloth Image for Classification:') #aciona o uploader
    st.set_option('deprecation.showfileUploaderEncoding', False)
    if file_uploader is not None: #Verifica erro no carregamento
        image=Image.open(file_uploader) #atribui a imagem à variável image
        text_io = io.TextIOWrapper(file_uploader)
        image=image.resize((180,180)) #redimensionar imagem
        st.image(image,'Uploaded image:') #faz o upload da imagem reduzida
           
        def classify_image(image,model):
            st.write("classifying......")
            img = ImageOps.grayscale(image)
            img=img.resize((28,28))
            if(add_selectbox=='Sequential'):
                img=np.expand_dims(img,0)
            else:
                img=np.expand_dims(img,0)
                img=np.expand_dims(img,3)
            img=(img/255.0) 
            img=1-img
            pred=model.predict(img)
            st.write("The Predicted image is:",class_names[np.argmax(pred)])
            st.write('Prediction probability :{:.2f}%'.format(np.max(pred)*100))
            
        st.write('Click for classify the image')
        if st.button('Classify Image'):
            if(add_selectbox=='Sequential'):
                st.write("You are choosen Image classification with Sequential Model")
                classify_image(image,seq_model)
                st.success('This Image successufully classified!')
                with st.spinner('Wait for it...'):
                    time.sleep(2)
                    st.success('Done!')
                    st.balloons()
                st.balloons()
            if(add_selectbox=='CNN'):
                st.write("You are choosen Image classification with CNN Model")
                classify_image(image,cnn_model)
                st.success('This Image successufully classified!')
                with st.spinner('Wait for it...'):
                    time.sleep(2)
                    st.success('Done!')
                    st.balloons()
    else:
        st.write("Please select image:")
