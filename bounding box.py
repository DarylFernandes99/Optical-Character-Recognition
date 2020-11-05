#Importing libraries
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt

text = ""

#Loading the model
classifier = load_model('D:/ML Project/letter(only).h5')

#Prediction classess
prediction = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

#Prediction function
def predict_letter(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (thresh, blackandWhiteImage) = cv2.threshold(~img_gray, 127, 255, cv2.THRESH_BINARY)
    
    #Fetching black and white image
    #img = ImageOps.invert(img)
    #img = img.convert('1')
    #Resizing image to (28, 28)
    blackandWhiteImage = cv2.resize(blackandWhiteImage, (128, 128))
    #image_copy = blackandWhiteImage.copy()
    #Convertign iamge to array
    blackandWhiteImage = np.array(blackandWhiteImage)
    #Reshaping image so that it can fit the CNN model
    blackandWhiteImage = blackandWhiteImage.reshape(1, 128, 128, 1)
    blackandWhiteImage = blackandWhiteImage /255.0
    
    result = classifier.predict(blackandWhiteImage)
    #print('Prediction: ' + prediction[np.argmax(result)])
    #print('Percentage: ' + "%.2f" % float(np.max(result)*100) + '%')
    #plt.imshow(img_copy)
    #print(prediction[np.argmax(result)])
    return prediction[np.argmax(result)]


#To convert words into sentences
def letter(roi, i, j):
    text_l = ""
    img_gray_letter = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    (thresh_letter, blackAndWhiteImage_letter) = cv2.threshold(img_gray_letter, 127, 255, cv2.THRESH_BINARY)
    img_dilate_letter = cv2.dilate(~blackAndWhiteImage_letter, np.ones((1, 1), np.uint8), iterations = 1)
    contours_letter, hierarchy_letter = cv2.findContours(img_dilate_letter, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_ctrs_letter = sorted(contours_letter, key=lambda ctr: cv2.boundingRect(ctr)[0], reverse=True)
    
    k = len(sorted_ctrs_letter)
    for contour_letter in sorted_ctrs_letter:
        (x,y,w,h) = cv2.boundingRect(contour_letter)
        #print('\t\tLetter: ', cv2.contourArea(contour_letter))
        if cv2.contourArea(contour_letter) > 10:
            roi2 = roi[y:y+h, x:x+w]
            path = "D:/ML Project/sentence/words/letter/roi-" + str(i) + "-" + str(j) + "-" + str(k) + ".png"
            cv2.imwrite(path, roi2)
            text_l = str(predict_letter(roi2)) + text_l
            k = k - 1
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return text_l

def words(roi, i):
    text_w = ""
    img_gray_word = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    (thresh_word, blackAndWhiteImage_word) = cv2.threshold(img_gray_word, 127, 255, cv2.THRESH_BINARY)
    img_dilate_word = cv2.dilate(~blackAndWhiteImage_word, np.ones((1, 1), np.uint8), iterations = 1)
    contours_word, hierarchy_word = cv2.findContours(img_dilate_word, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_ctrs_word = sorted(contours_word, key=lambda ctr: cv2.boundingRect(ctr)[0], reverse=True)
    
    j = len(sorted_ctrs_word)
    for contour_word in sorted_ctrs_word:
        (x,y,w,h) = cv2.boundingRect(contour_word)
        #print('\tWords: ', cv2.contourArea(contour_word))
        if cv2.contourArea(contour_word) > 10:
            roi1 = roi[y:y+h, x:x+w]
            text_w = str(letter(roi1, i, j)) + text_w
            path = "D:/ML Project/sentence/words/roi-" + str(i) + "-" + str(j) + ".png"
            cv2.imwrite(path, roi1)
            j = j - 1
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    text_w = text_w + "\t"
    return text_w


#Reading image
img = cv2.imread('D:/ML Project/test1.png')

if img.shape[1] > 1000:
    scale_percent = 30 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, cv2.INTER_AREA)

img_copy = img.copy()

#COnvert to grayscale and blurring the image
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
(thresh, blackAndWhiteImage) = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

#Detecting edges
#edges = cv2.Canny(~blackAndWhiteImage, 10, 100)

#Finding contours
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (14, 1))
img_dilation = cv2.dilate(~blackAndWhiteImage, rect_kernel, iterations = 1)

contours, hierarchy = cv2.findContours(img_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

#Sorting contours from left to right andtop to bottom
sorted_ctrs = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0] + cv2.boundingRect(ctr)[1] * img.shape[1] )


i = 0
for contour in sorted_ctrs:
    (x,y,w,h) = cv2.boundingRect(contour)
    #print(cv2.contourArea(contour))
    if (cv2.contourArea(contour) > 10): #& (cv2.contourArea(contour) < 9000):
        roi = img_copy[y:y+h, x:x+w]
        path = "D:/ML Project/sentence/roi-" + str(i) + ".png"
        cv2.imwrite(path, roi)
        text = text + str(words(roi, i)) + "\n"
        i = i + 1
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

print("Converted Text")
print(text)
#cv2.drawContours(img_dilation, contours, -1, (255, 0, 0), 3)