from fastai import *
from fastai.vision import *


import os
import numpy as np


#Making a Databunch from the file 
path = ('C:/Users/Vikas/Desktop/Flask/flowers')
data = ImageList.from_folder(path)
data2 = data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,
        ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)

print(data2.classes)

#Loading the model

classes = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

learn = cnn_learner(data2, models.resnet34)
learn.load('C:/Users/Vikas/Desktop/Flask/models/stage-1')

img = open_image('C:/Users/Vikas/Desktop/Flask/rose.jpg')

pred_class,pred_idx,outputs = learn.predict(img)
print(pred_class)

