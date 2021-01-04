import cv2
import csv
import numpy as np
import sys
from skimage import io, color, img_as_ubyte
from skimage.feature import greycomatrix, greycoprops
from sklearn.metrics.cluster import entropy
import pdb
import os
from scipy.stats import skew
import imutils

def doThis(filenya,labelnya, namafile):
    image = cv2.imread(filenya)
    
    #Fitur bentuk 7
    image2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    humoment = cv2.HuMoments(cv2.moments(image2)).flatten()
    #------------------------------------------------------

    #fitur tekstur 48
    grayImg = img_as_ubyte(color.rgb2gray(image))

    distances = [1, 2, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    properties = ['energy', 'dissimilarity', 'contrast', 'homogeneity']
    
    glcm = greycomatrix(grayImg, 
                        distances=distances, 
                        angles=angles,
                        symmetric=True,
                        normed=True)
    
    feats = np.hstack([greycoprops(glcm, prop).ravel() for prop in properties])
    #---------------------------------------------------------------------------
    
    #fitur warna 9
    red, green, blue = cv2.split(image)
    fr, frsd, varr = np.mean(red), np.std(red), np.var(red)
    fg, fgsd, varg = np.mean(green), np.std(green), np.var(green)
    fb, fbsd, varb = np.mean(blue), np.std(blue), np.var(blue)
    
    warna = np.array([fr, frsd, fg, fgsd,fb, fbsd, varr, varg, varb])
    #-----------------------------------------------


    feats = np.concatenate((feats, warna), axis=0)
    feats = np.concatenate((feats, humoment), axis=0)
    datafitur = list(feats)
    datafitur.append(labelnya)
    
    dataSet = "hewan1.csv"
    with open(dataSet, "a") as f :
        writer = csv.writer(f)
        writer.writerow(datafitur)
    return

path = "./Panda"
label = input("Label Gambar ? ")
for file in os.listdir(path):
    current_file = os.path.join(path, file)
    doThis(current_file, label, file)
    print(current_file)
