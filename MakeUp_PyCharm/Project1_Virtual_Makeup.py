#!/usr/bin/env python
# coding: utf-8
import os
from http.server import BaseHTTPRequestHandler
from http.server import HTTPServer
import time
from urllib.parse import urlparse, parse_qs
import json
import numpy as np
from flask import Flask
import pyodbc
import ast
import pandas as pd

hostName = "127.0.0.1"
hostPort = 9007


class MyServer(BaseHTTPRequestHandler):
    app = Flask(__name__)

    def sentiment(self):
        pass
    #     do something
    # to use the model you need firefly

    def do_GET(self):
        # getparams
        query_components = parse_qs(urlparse(self.path).query)
        self.send_response(200)
        self.send_header("Content-type", 'application/json')
        self.end_headers()
        if "sentiment" in self.path:
            pass
        #     call function
        self.send_response(200)
        self.end_headers()
        json_content = json.dumps("the function/ model response", ensure_ascii=False)  # json.dumps(res)
        print(json_content)
        self.wfile.write(bytes(str(json_content), "utf-8"))
        return

# generate the server
myServer = HTTPServer((hostName, hostPort), MyServer)
print(time.asctime(), "Server Starts - %s:%s" % (hostName, hostPort))
try:
    myServer.serve_forever()
except KeyboardInterrupt:
    pass

# stop the server
myServer.server_close()
print(time.asctime(), "Server Closed - %s:%s" % (hostName, hostPort))

# # <font color="blue">Project1: Virtual Makeup </font>
# We have already seen interesting applications of facial features and landmarks such as aging, beardify, face swap etc.
#
# In this project, you will build features for a Virtual Makeup application! Given below are a few features that are required to be built in the application.
#
# 1. Apply Lipstick
# 1. Apply Blush
# 1. Apply Eye-Liners or Eye-Lashes
# 1. Apply Glasses
# 1. Apply Eye Color ( Lens Color)
# 1. Apply Hat/Cap
# 1. Apply Ear-Rings
# 1. Change Hair Color
#
# ### <font color="green">Your Task</font>
# Implement any 2 features from the list above
#
# We have provided a sample image. You can use your own image for experimentation as well as come up with other interesting features.
#
# ### <font color="green">Submission</font>
# Once you are done, you have to create a video, explaining the main part of the code, upload it to youtube or any other video sharing service and provide the link in the form given in the submission section of the course.
#
# ### <font color="green">Marks Distribution</font>
#
# 1. Feature 1: 35 marks
# 1. Video for Feature 1: 15 marks
# 1. Feature 2: 35 marks
# 1. Video for Feature 2: 15 marks

# In[1]:

import cv2, sys, dlib, time, math
import numpy as np
import matplotlib.pyplot as plt
import os # relative path reading
import colorsys
import matplotlib
# get_ipython().run_line_magic('matplotlib', 'inline')

#print(cv2.__version__)
#print(dlib.__version__)
#print(np.__version__)
#print(matplotlib.__version__)


# In[2]:


# for google drive ipynb
# from google.colab import drive
# drive.mount('/content/gdrive/')


# In[3]:


# set the directory path to the project folder
#from tensorflow import keras

path = r'M:\BinaProj\repozitory\Virtual-Makeup-Product\MakeUp_PyCharm'
os.chdir(path)
#print(os.getcwd())


# Load faceBlendCommon file to use common functions.


# load the faceBlendCommon script
import faceBlendCommon as fbc


# In[5]:


import matplotlib
matplotlib.rcParams['figure.figsize'] = (8.0,8.0)
matplotlib.rcParams['image.cmap'] = 'gray'
matplotlib.rcParams['image.interpolation'] = 'bilinear'


# # <font color="blue">Load landmark detector</font>
# We have provided code for loading the model for landmark detector.
#
# Load the given 68 point landmark detector

# In[6]:


# Landmark model location
PREDICTOR_PATH =  "shape_predictor_68_face_landmarks.dat"

# Get the face detector
faceDetector = dlib.get_frontal_face_detector()
# The landmark detector is implemented in the shape_predictor class
landmarkDetector = dlib.shape_predictor(PREDICTOR_PATH)


# # <font color="blue">Read Image</font>
# We load the image and convert it to RGB format so that it can be used by Dlib and also for displaying using matplotlib.
#
# ### <font color="red">You can also use any other image.</font>

# In[7]:


no_makeup = cv2.imread("girl-no-makeup.png")

imDlib = cv2.cvtColor(no_makeup,cv2.COLOR_BGR2RGB)
plt.imshow(imDlib)


# In[8]:


# For reference, I'll be using two images
lipstick = cv2.imread("girl-no-makeup.png")

imDlib2 = cv2.cvtColor(lipstick,cv2.COLOR_BGR2RGB)
plt.imshow(imDlib2)


# # <font color="blue">Calculate Landmarks</font>
# We are providing code for calculating the landmarks in this section. So that you can focus on the feature implementation.

# In[9]:


# points = fbc.getLandmarks(faceDetector, landmarkDetector, imDlib)
# print(points)


# # <font color="blue">TODO</font>
# Implementation of 2 features:
#

# Below code is used as a visual indicator for which points belong to which facial feature
#

# In[10]:


choice = imDlib2
imcopy = choice.copy()

points = fbc.getLandmarks(faceDetector, landmarkDetector, choice)
# print(points)

# feature point reveal as a visual reference
for p in points[0:17]:
    # Jaw points
    cv2.circle(imcopy, p, 4, (0,0,255), thickness=-1)
for p in points[18:27]:
    # eyebrows points
    cv2.circle(imcopy, p, 4, (255,0,0), thickness=-1)
for p in points[28:36]:
    # nose points
    cv2.circle(imcopy, p, 4, (0,255,0), thickness=-1)
for p in points[36:48]:
    # eyes points
    cv2.circle(imcopy, p, 4, (0,255,255), thickness=-1)
for p in points[48:]:
    # lips points
    cv2.circle(imcopy, p, 4, (255,0,255), thickness=-1)
# plt.imshow(imcopy)
plt.imshow(imcopy)



# # <font color="blue">Feature 1: Lip Change</font>
# Write code for the first feature you want to implement from the above list. Display the output image after writing the code.

# # Choose a color and Image

# In[11]:


# Change this color to change lipstick color!
color = (255, 0, 128)

# Change the Choice of image here
choice = imDlib2
imcopy = choice.copy()

points = fbc.getLandmarks(faceDetector, landmarkDetector, choice)


# Reference landmarks
#
# ![](https://www.researchgate.net/publication/327500528/figure/fig9/AS:668192443748352@1536320901358/The-ibug-68-facial-landmark-points-mark-up.ppm)

# In[12]:


# Get points of the lips

# I will be using cv2.pollyFill to build the mask, polyfill takes in vectorized set of points
# this means a line will be drawn between point[0] -> point[1] -> point[2] -> ... -> point[N] -> point[0] and subsequently filled

# So this means how upperlips and lowerlips are ordered MATTERS or else the mask will be drawn incorrectly
upperlips = points[48:55] + points[60:65][::-1]
lowerlips = points[48:49] + points[54:60][::-1] + points[64:]


# In[13]:


# visualize the points
imcopy = choice.copy()
imcopy2 = choice.copy()

for p in upperlips:
    cv2.circle(imcopy, p, 4, (0,0,255), thickness=-1)

plt.imshow(imcopy)


# In[14]:


for p in lowerlips:
    cv2.circle(imcopy2, p, 4, (255,0,0), thickness=-1)
plt.imshow(imcopy2)


# # Creating the lips mask

# In[15]:


# Version 2 of lip mask

# cv2.pollyFill wants np.arrays to be passed to it. Currently upperlips and lowerlips are a list(tuples)

# They need to be converted from list(tuples) to list(list(int))

print('Original list of tuples')
print(upperlips)
print('\n')

uHull = [[p[0],p[1]] for p in upperlips]
# for p in upperlips:
#   uHull.append([p[0], p[1]])
lHull = [[p[0],p[1]] for p in lowerlips]
# for p in lowerlips:
#   lHull.append([p[0], p[1]])

print('Converted into list of lists')
print(uHull)
print('\n')

uHull = np.array(uHull)
lHull = np.array(lHull)

print('Converted into numpy arrays')
print(uHull)
print('\n')


# In[16]:


# We build the mask for the lips
row, col, _ = choice.shape
mask = np.zeros((row, col), dtype=choice.dtype)

cv2.fillPoly(mask, [uHull], (255));
cv2.fillPoly(mask, [lHull], (255));

bit_mask = mask.astype(bool)


# In[17]:


plt.imshow(bit_mask)


# In[18]:


# Find bounding box for mask preview
lst = upperlips + lowerlips
xmin, xmax = min(lst, key = lambda i : i[1])[1], max(lst, key = lambda i : i[1])[1]
ymin, ymax = min(lst, key = lambda i : i[0])[0], max(lst, key = lambda i : i[0])[0]


# In[19]:


# Up-close view of the mask
plt.imshow(bit_mask[xmin - 5:xmax + 5, ymin - 5:ymax + 5])


# In[ ]:


pixel = np.zeros((1,1,3), dtype=np.uint8)
r_ = 0
g_ = 1
b_ = 2

pixel[:,:,r_], pixel[:,:,g_], pixel[:,:,b_] = color[r_], color[g_], color[b_]


# In[50]:


plt.imshow(pixel)


# In[48]:


out = choice.copy()

# Convert image of person from RGB to HLS
pixel_hsl = cv2.cvtColor(pixel, cv2.COLOR_RGB2HLS)
outhsv = cv2.cvtColor(out,cv2.COLOR_RGB2HLS)

channel = 0

# extract the hue channels
hue_img = outhsv[:,:,channel]
hue_pixel = pixel_hsl[:,:,0]

hue_img[bit_mask] = hue_pixel[0,0]

out = cv2.cvtColor(outhsv,cv2.COLOR_HLS2RGB)


# In[ ]:


plt.imshow(out)


# In[24]:


plt.imsave('girl-no-makeup-'+'lip-color-change.png', out)


# # <font color="blue">Feature 2</font>
# Write code for the second feature you want to implement from the above list. Display the output image after writing the code.

# In[25]:


# !pip uninstall keras
# !pip install keras
# !pip uninstall tensorflow
# !pip install tensorflow


# In[26]:


import glob
import time



# In[27]:


# print(keras.__version__)


# # Helper Method

# In[28]:


# Modified code taken from author
# def predict(image, height=224, width=224):
#     im = image.copy()
#     im = im / 255
#     im = cv2.resize(im, (height, width))
#     im = im.reshape((1,) + im.shape)
#
#     pred = model.predict(im)
#     mask = pred.copy()
#     mask = mask.reshape((224, 224,1))
#     row, col, _ = image.shape
#     mask = cv2.resize(mask, (col, row))
#     return mask


# Using a convolutional network to predict the hairmask model

# In[29]:


# load hair detection model
# model source https://github.com/thangtran480/hair-segmentation/releases
# model = keras.models.load_model('C:/Users/212314363/hairnet_matting.hdf5')
#
#
# # In[30]:
#
#
# img = choice.copy()
#
# # Predict the mask from the image
# hairmask = predict(img)
#
#
# # Create the mask from the predicted model. The mask goes through some conversions so that it matches the image's dimensions
#
# # In[31]:
#
#
# ## The next few steps reshapes the predicted mask to a 3 dimensional image
# print("Original shape")
# print(img.shape)
#
# print("predicted mask shape")
# print(hairmask.shape)
#
#
#
# # # add a new dimensions to the original mask
# # reshaped_hairmask = hairmask[:,:,np.newaxis]
#
# # print("mask additiona new dimension")
# # print(reshaped_hairmask.shape)
#
#
# # row, col, _ = reshaped_hairmask.shape
# # reshaped_hairmask = reshaped_hairmask.repeat(3, axis = 2)
# # print("final mask shape")
# # print(reshaped_hairmask.shape)
#
#
# # Mask Preview
#
# # In[32]:
#
#
# plt.imshow(hairmask)
#
#
# # In[33]:
#
#
# # Mask Creation
# threshold = 0.7
# bit8_hairmask = hairmask.copy()
#
# # Convert the float hairmask into uint8 values
# bit8_hairmask[bit8_hairmask > threshold] = 255
# bit8_hairmask[bit8_hairmask <= threshold] = 0
#
# # convert unint8 mask to a boolean mask
# bin_hairmask = bit8_hairmask.astype(np.bool)
# print(bin_hairmask.shape)
# plt.imshow(bin_hairmask)
#
#
# # In[34]:
#
#
# # Convert the 8bit mask into a 3 dimensional image with a hue swap
# rgb_mask = bit8_hairmask.copy()
#
# # add new dimension
# rgb_mask = rgb_mask.astype(np.uint8)
# rgb_mask = rgb_mask[:,:,np.newaxis]
# print(rgb_mask.shape)
#
# # repeat the dimension in the 3rd axis
# rgb_mask = rgb_mask.repeat(3, axis = 2)
#
# print(rgb_mask.shape)
#
#
# # In[35]:
#
#
# # break the color into its numerical components
# r, g, b = color
#
# # Set the r g b channels to their set
# rgb_mask[:,:,0][bin_hairmask] = r
# rgb_mask[:,:,1][bin_hairmask] = g
# rgb_mask[:,:,2][bin_hairmask] = b
#
#
# # In[36]:
#
#
# plt.imshow(rgb_mask)
#
#
# # In[37]:
#
#
# # RGB to HLS Conversion
# hls_mask = cv2.cvtColor(rgb_mask, cv2.COLOR_RGB2HLS)
# hls_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
# channel = 0
#
# h_mask, h_img = hls_mask[:,:, channel], hls_img[:,:, channel]
#
# h_img[bin_hairmask] = h_mask[bin_hairmask]
#
# rgb_img = cv2.cvtColor(hls_img, cv2.COLOR_HLS2RGB)
#
#
# # In[38]:
#
#
# plt.imshow(rgb_img)
#
#
# # In[39]:
#
#
# plt.imsave('girl-no-makeup-2'+'hair-color-change.png', rgb_img)
#
#
# # In[39]:
#
#
#

# In[4]:
#
