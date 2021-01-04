import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import tree
import warnings
import sys
import cv2
from skimage import io, color, img_as_ubyte
from skimage.feature import greycomatrix, greycoprops
from sklearn.metrics.cluster import entropy
from scipy.stats import skew

warnings.filterwarnings('ignore')

#droppedFile = sys.argv[1]
#print droppedFile
#filenya =  droppedFile
#pdb.set_trace()

images = ['./TEST/test1.jpg','./TEST/test2.jpg',
          './TEST/test3.jpg', './TEST/test4.jpg','./TEST/test5.jpg','./TEST/test6.jpg']

for x in images:
    image = cv2.imread(x)

    red, green, blue = cv2.split(image)
    image2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #threshold dulu baru HU's Momment
    ret,image3 = cv2.threshold(image2,127,255,cv2.THRESH_BINARY)

    #Fitur bentuk 7
    #humoment = cv2.HuMoments(cv2.moments(image2)).flatten()
    humoment = cv2.HuMoments(cv2.moments(image3)).flatten()

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

    #fitur warna 9
    red, green, blue = cv2.split(image)
    fr, frsd, varr = np.mean(red), np.std(red), np.var(red)
    fg, fgsd, varg = np.mean(green), np.std(green), np.var(green)
    fb, fbsd, varb = np.mean(blue), np.std(blue), np.var(blue)
    
    ciriwarna = np.array([fr, frsd, fg, fgsd,fb, fbsd, varr, varg, varb])

    feats = np.concatenate((feats, ciriwarna), axis=0)
    feats = np.concatenate((feats, humoment), axis=0)
    datafitur = list(feats)  

    data = pd.read_csv('HEWAN1.csv')

    y = data.KELAS
    X = data.drop('KELAS', axis=1)

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, y)

    X_test = np.array(datafitur).reshape(1, -1)
    print(X_test)
    y_predict = clf.predict(X_test)
    print(y_predict)

    #y_predict2 = clf.predict_proba(X_test)
    #proba = max(y_predict2[0,i] for i in range(1))*100
    #print "Probrabilitas = " + str(proba) + "%"
