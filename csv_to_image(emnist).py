import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image,ImageOps
import cv2
import glob

values =  ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
           'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
           'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
           'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 
           'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 
           'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

dataset = pd.read_csv('C:/Users/daryl/Desktop/ip/emnist/emnist-byclass-test.csv', header = None)

label = dataset.iloc[:,:1].values
images = dataset.iloc[:,1:].values

n = len(label)

#COnvert csv rows to image and write to folder
for i in range(n):
    path = "D:/ML Project/Dataset/test_set/"
    lbl = label[i]
    lbl = values[int(lbl)]
    img = images[i]
    img = img.reshape((28,28))
    asci = ord(lbl)
    if asci <= 90 and asci >= 65:
        path += "U"
    path+=lbl+"/"+lbl+str(i)+".png"
    print(path, str(i), sep = '\n')
    cv2.imwrite(path,img)
    

#To flip and rotate the images in the directory(images are in flipped and rotated format)
for ch in values:
    path = "D:/ML Project/Dataset/test_set/"
    asci = ord(ch)
    if asci <= 90 and asci >= 65:
        path += "U"
    path += ch + "/"+ "*.*"
    for file in glob.glob(path):
        img = Image.open(file)
        img = ImageOps.flip(img)
        img = img.transpose(Image.ROTATE_270)
        file = file.replace("/", "\\")
        img.save(file)
        print(file)

#To get number of images present in directory
import os
cpt = sum([len(files) for r, d, files in os.walk("D:/ML Project/Dataset/train_set/")])
