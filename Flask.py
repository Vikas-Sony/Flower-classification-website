# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 09:50:18 2019

@author: Vikas
"""
from fastai import *
from fastai.vision import *


import os
import numpy as np


#Making a Databunch from the file 
path1 = ('C:/Users/Vikas/Desktop/Flask/flowers')
data = ImageList.from_folder(path1)
data2 = data = ImageDataBunch.from_folder(path1, train=".", valid_pct=0.2,
        ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)

print(data2.classes)

#Loading the model

classes = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

learn = cnn_learner(data2, models.resnet34)
learn.load('C:/Users/Vikas/Desktop/Flask/models/stage-1')

img = open_image('C:/Users/Vikas/Desktop/Flask/rose.jpg')
path2 = ('C:/Users/Vikas/Desktop/Flask/templates/uploads')

pred_class,pred_idx,outputs = learn.predict(img)
print(pred_class)

from flask import Flask, render_template, request, redirect
app = Flask(__name__)

@app.route("/")
def home():
	return render_template('index.html')

@app.route("/about/")
def about():
	return render_template('about.html')	
@app.route("/upload-image",methods=["GET", "POST"])
def upload_image():

    if request.method == "POST":

        if request.files:

            image = request.files["image"]
            img = open_image(image)
            image.save(os.path.join(path2, image.filename))
            

            print(image.filename)
            

            

            pred_class,pred_idx,outputs = learn.predict(img)
            print(pred_class)
            # print(path2/image.filename)

            return render_template("Upload_image.html", Prediction_text = pred_class, url1 = image.filename)


    return render_template("Upload_image.html")	




if __name__ == '__main__':
	app.run(debug = True)

