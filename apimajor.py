#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 22:01:26 2019

@author: dipesh
"""

from flask import Flask,jsonify,request
from flask_cors import CORS
import cv2
import numpy 
from skimage.transform import rescale
from keras.models import load_model

app = Flask(__name__)
cors = CORS(app) 

@app.route('/',methods=['GET'])
def get():
    return "helllo"
    

@app.route('/image',methods=['POST'])
def receive_image():
#    return "receiving image"
    name = request.form['name']
    email = request.form['email']
    filestr = request.files['file_cv'].read()
    npimg = numpy.fromstring(filestr, numpy.uint8)
    img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
   
    img = rescale(img, 1./255, anti_aliasing=False)
    img = cv2.resize(img, (224,224))
    img = numpy.expand_dims(img, axis = 0)
    
    mod=load_model('model.hd5')
    predicted = mod.predict(img)
    print(predicted)
    y_pred = predicted[0][0] > 0.5
    print(y_pred)
    percent_chance = round(predicted[0][0]*100, 2)
    print(percent_chance)
    return jsonify({"name":name,"email":email,"percent":percent_chance,"detection":int(y_pred)})
    
app.run(port=8090)
