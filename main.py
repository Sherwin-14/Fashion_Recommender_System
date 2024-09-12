import streamlit as st
import os 
import tensorflow as tf
import pickle
import numpy as np

from PIL import Image
from numpy.linalg import norm
from annoy import AnnoyIndex
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

input_shape =  (None,224,224,3)

model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

feature_list = np.array(pickle.load(open('imp/embeddings.pkl','rb')))
filenames = pickle.load(open('imp/filenames.pkl','rb'))

st.title('Fashion Recommender System')

def save_uploaded_file(uploaded_file):
    try :
        with open(os.path.join('uploads',uploaded_file.name),"wb") as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0  

def feature_extraction(img_path,model):
    img = image.load_img(img_path,target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array,axis=0)
    preprocessed_image = preprocess_input(expanded_img_array)
    result  = model.predict(preprocessed_image).flatten()
    normalized_result = result/ norm(result)
    
    return normalized_result

def recommend(features,feature_list):

    index = AnnoyIndex(features.shape[0])# Length of item vector that will be indexed

    for i,vector in enumerate(feature_list):
        index.add_item(i,vector)    

    index.build(n_trees=10)# 10 trees
    k = 7

    nearest_neighbors = index.get_nns_by_vector(features, k)

    return nearest_neighbors
 
# steps
# file upload -> save

uploaded_file = st.file_uploader("Choose an Image")

if uploaded_file is not None:
   if save_uploaded_file(uploaded_file):
       display_image = Image.open(uploaded_file)
       col1,col2,col3 = st.columns(3)

       with col2:
            st.image(display_image, use_column_width=True)
        # load file -> feature extract
       features = feature_extraction(os.path.join('uploads',uploaded_file.name),model)
        # recommendation
       nearest_neighbors = recommend(features,feature_list)
        # show 
       col1,col2,col3,col4,col5 = st.columns(5)

       with col1:
           st.image(filenames[nearest_neighbors[0]],use_column_width=True) 
       
       with col2:
           st.image(filenames[nearest_neighbors[1]],use_column_width=True)  

       with col3:
           st.image(filenames[nearest_neighbors[2]],use_column_width=True)   

       with col4:
           st.image(filenames[nearest_neighbors[3]],use_column_width=True)    

       with col5:
           st.image(filenames[nearest_neighbors[4]],use_column_width=True)                

   else:
       st.header("Some error occurred in file upload")    
