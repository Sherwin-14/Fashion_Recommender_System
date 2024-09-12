import tensorflow as tf
import keras
import numpy as np
import os
import pickle
from PIL import Image
from numpy.linalg import norm
from tqdm import tqdm
from keras import ops
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

input_shape =  (None,224,224,3)

model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

#model.build(input_shape) 
#model.summary()

def extract_features(img_path,model):
    img = image.load_img(img_path,target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array,axis=0)
    preprocessed_image = preprocess_input(expanded_img_array)
    result  = model.predict(preprocessed_image).flatten()
    normalized_result = result/ norm(result)

    return normalized_result


filenames = []

for file in os.listdir('data/images'):
    filenames.append(os.path.join('data/images',file))


feature_list = []

for file in tqdm(filenames):
    feature_list.append(extract_features(file,model))

pickle.dump(feature_list,open('embeddings.pkl','wb'))
pickle.dump(filenames,open('filenames.pkl','wb'))

