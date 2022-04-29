"""
MODEL DE SEGMENTATION D'IMAGES (pour la prédiction)

Ce script présente un modèle pour segmenter les images RGB. Il est composé de 4 parties :
1- Importation des librairies
2- Importation de l'image au format RGB [Input]
3- Traitement de l'image par le modèle segmentation [Processing]
4- Exporation de l'image segmentée au format binaire [Output]

En particulier l'étape de traitement (étape 3) est composée en trois sous-parties :
i) Redimension de l'image en 256 x 256 pixel
ii) Segmentation de l'image par l'algorithme U-NET
iii) Application du seuil de segmentation optimisé sur l'image
"""

import sys

#Etape 1
import numpy
import tensorflow as tf
from tensorflow import keras
#!pip install focal_loss
from focal_loss import BinaryFocalLoss
from tensorflow.keras import backend as K
import numpy as np
import cv2
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
import os

def redimension(image):
    X = np.zeros((1,256,256,3),dtype=np.uint8)
    img = cv2.imread(image,cv2.COLOR_BGR2RGB)
    size_ = img.shape
    img = img[:,:,:3]
    X[0] = resize(img, (256, 256), mode='constant', preserve_range=True)
    return X,size_

def dice_coefficient(y_true, y_pred):
    eps = 1e-6
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + eps) #eps pour éviter la division par 0 



if __name__ == "__main__":
    #Etape 2 : Input
    image_sample = os.listdir("./input")
    sys.stdout.write("Input done\n")
    #Etape 3 : i) Redimension de l'image en 256x256 pixel
    image_sample_reshaped = []
    size_original = []
    for sample in image_sample:
        image_reshaped,size_ = redimension("./input/"+sample)
        image_sample_reshaped.append(image_reshaped)
        x,y,z = size_
        size_original.append((x,y))
    sys.stdout.write("Reshaped done\n")
    #Etape 3 : ii) Segmentation de l'image par l'algorithme U-NET pré-entrainé
    model_new = tf.keras.models.load_model("best_model_FL_BCE_0_5.h5",custom_objects={'dice_coefficient': dice_coefficient})
    sys.stdout.write("Model imported\n")    
    image_prediction = []
    for sample in image_sample_reshaped:
        prediction = model_new.predict(sample)
        
        #Application du seuil de segmentation optimisé
        preds_test_t = (prediction > 0.30000000000000004)
        image_prediction.append(preds_test_t)
    sys.stdout.write("Prediction done\n")
    #Etape 4 : Output
    image_output = []
    for sample,i,name in zip(image_prediction,range(len(size_original)),image_sample):
        preds_test_t = resize(sample[0,:,:,0], size_original[i], mode='constant', preserve_range=True)
        image_output.append(preds_test_t)
        cv2.imwrite("./output/output_"+name,preds_test_t*255)
    sys.stdout.write("Export done\n")