
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

import math
import numpy as np
import argparse
import glob
import cv2
import sys
import os


# define Enum class
class Enum(tuple): __getattr__ = tuple.index

global Total
Total = 0
# Enumerate material types for use in classifier
Coin = Enum(('Five', 'Two', 'One', 'no'))



def calcHistogram(img):
    # create mask
    m = np.zeros(img.shape[:2], dtype="uint8")
    (w, h) = (int(img.shape[1] / 2), int(img.shape[0] / 2))
    cv2.circle(m, (w, h), 60, 255, -1)

    # calcHist expects a list of images, color channels, mask, bins, ranges
    h = cv2.calcHist([img], [0, 1, 2], m, [8, 8, 8], [0, 256, 0, 256, 0, 256])

    # return normalized "flattened" histogram
    return cv2.normalize(h, h).flatten()


def predictMaterial(roi):
    # calculate feature vector for region of interest
    hist = calcHistogram(roi)
    # predict material type
    pickle_in = open("clf.pickle","rb")
    vec = pickle.load(pickle_in)
    clf = vec[0]
    s = clf.predict([hist])
    #print(Coin[int(s)])
    # return predicted material type
    return Coin[int(s)]



def test(path):
        global Total
        image = cv2.imread(path)
        i = 0
        # resize image while retaining aspect ratio
        d = 1024 / image.shape[1]
        dim = (1024, int(image.shape[0] * d))
        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


        # create a copy of the image to display results
        output = image.copy()
        # convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        height,width = gray.shape
        mask = np.zeros((height,width), np.uint8)
        # improve contrast accounting for differences in lighting conditions:
        # create a CLAHE object to apply contrast limiting adaptive histogram equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        # circles: A vector that stores x, y, r for each detected circle.
        # src_gray: Input image (grayscale)
        # CV_HOUGH_GRADIENT: Defines the detection method.
        # dp = 2.2: The inverse ratio of resolution
        # min_dist = 100: Minimum distance between detected centers
        # param_1 = 200: Upper threshold for the internal Canny edge detector
        # param_2 = 100*: Threshold for center detection.
        # min_radius = 50: Minimum radius to be detected.
        # max_radius = 120: Maximum radius to be detected.
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=2.2, minDist=270,
                                   param1=200, param2=100, minRadius=50, maxRadius=200)
        print(circles)
        print("Entered")

        # todo: refactor
        diameter = []
        materials = []
        coordinates = []

        count = 0   
        if circles is not None:
            # append radius to list of diameters (we don't bother to multiply by 2)
            for (x, y, r) in circles[0, :]:
                diameter.append(r)

            # convert coordinates and radii to integers
            circles = np.round(circles[0, :]).astype("int")

            # loop over coordinates and radii of the circles
            for (x, y, d) in circles:
                count += 1
                
                # add coordinates to list
                coordinates.append((x, y))
                
                # extract region of interest
                roi = image[y - d:y + d, x - d:x + d]
                
                material = predictMaterial(roi)
                print(material)
                if(material == "Five"):
                    Total+=5
                elif(material == "Two"):
                    Total+=2
                elif(material == "One"):
                    Total+=1
                elif(material == "no"):
                    Total+=0
                m = np.zeros(roi.shape[:2], dtype="uint8")
                w = int(roi.shape[1] / 2)
                h = int(roi.shape[0] / 2)
                cv2.circle(m, (w, h), d, (255), -1)
                maskedCoin = cv2.bitwise_and(roi, roi, mask=m)
                '''Output folder name inplace of New_Data'''
                file_name = "New_Data" + "/" + str(i) + ".png"
                cv2.imwrite(file_name.format(count), maskedCoin)
                i = i + 1
            print("The total is ", Total)
            return Total
#Call This Function
total_val = test("Input/image.jpg")
