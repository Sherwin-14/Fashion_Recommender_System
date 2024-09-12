import pickle
import tensorflow as tf
import numpy as np
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from annoy import AnnoyIndex
import random

feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

input_shape =  (None,224,224,3)

model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

img = image.load_img('data/images/1164.jpg',target_size=(224,224))
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array,axis=0)
preprocessed_image = preprocess_input(expanded_img_array)
result  = model.predict(preprocessed_image).flatten()
normalized_result = result/ norm(result)

index = AnnoyIndex(normalized_result.shape[0])# Length of item vector that will be indexed

for i,vector in enumerate(feature_list):
    index.add_item(i,vector)    

index.build(n_trees=10)# 10 trees
k = 5 

# getting indices of nearest neighbors
nearest_neighbors = index.get_nns_by_vector(normalized_result, k)

print("Indices of nearest neighbors:", nearest_neighbors)

for i in nearest_neighbors:
    print("Image at index", i, ":", filenames[i])




