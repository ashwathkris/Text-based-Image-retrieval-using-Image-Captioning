from flask import Flask, redirect, url_for, request, render_template, send_from_directory
import nltk
import ssl
# nltk.download('punkt')
from rank_bm25 import BM25Okapi
import os
import json
import string
import numpy as np
import pandas as pd
from numpy import array
from pickle import load
from PIL import Image
import pickle
from collections import Counter
import matplotlib.pyplot as plt
import sys, time, warnings
warnings.filterwarnings("ignore")
from nltk.corpus import stopwords
from nltk import download
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi

app = Flask(__name__, static_folder = "Flicker8k_Dataset")
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
image_names = []

import logging
logging.basicConfig(level=logging.DEBUG)

@app.route("/", methods=['POST', 'GET'])
def temp():
    logging.debug("Accessed home page")
    return render_template("home.html")

f = open('data.json',)
# returns JSON object as a dictionary
data = json.load(f)


@app.route("/renderupload", methods=['POST','GET'])
def showuploadpage():
    return render_template('upload.html')

@app.route("/upload", methods = ['POST'])
def upload():
    target = os.path.join(APP_ROOT, 'Flicker8k_Dataset/')
    print(target)
    if not os.path.isdir(target):   #if folder does not exist
        os.mkdir(target)

    for file in request.files.getlist("file"):
        print(file)
        filename = file.filename
        destination = "/".join([target,filename])
        print(destination)
        file.save(destination)
        image_id = destination 
        #image_id = filename 
        #image_id = APP_ROOT+'gallery\\' + filename
        result = evaluate(image_id) #generates caption
        for i in result:
            if i=="<unk>":
                result.remove(i)
        result_join = ' '.join(result)
        result_final = result_join.rsplit(' ', 1)[0]
        print ('Prediction Caption:', result_final)
        predicted_caption = result_final
        #abs_id = '../gallery\\' + filename                        #changed later
        image_names.append(filename)
        image_id = filename
        tag=request.form["tag"]
        new_pic = {"id" : image_id, "caption":[predicted_caption],"tags":[tag]}
        def write_json(data, filename='data.json'):
            with open(filename,'w') as f:
                json.dump(data, f, indent=4)
        with open('data.json') as json_file:
            data = json.load(json_file)
            temp = data['pics']
            # appending data 
            temp.append(new_pic)
        write_json(data)
        #file.save(destination)
    return render_template("complete.html")

@app.route('/rendersearch', methods=['GET'])
def home():
    logging.debug("Rendering search.html")
    return render_template('search.html', image='searchicon.png')


from rank_bm25 import BM25Okapi

@app.route('/search',methods=['POST','GET'])
def query():
    if request.method=='POST':
        query=request.form['query']
        # query_tag=request.form['searchtag']
        
        tagged_images=list()

        # for i in data['pics']:                          #to check for images with matching tag, and storing their ids in tagged_images
        #     for j in i['tags']:
        #         if query_tag == j:
        #             tagged_images.append(i['id'])       #getting list of captions
        list_of_caption = list()
        if len(tagged_images)==0:                       #if there are no images with matching tag
            for i in data['pics']:
                for j in i['caption']:
                    list_of_caption.append(j)
        else:
            for i in data['pics']:
                if i['id'] in tagged_images:
                    for j in i['caption']:
                        list_of_caption.append(j)
        #print(list_of_caption)
        tok_text = list()   #tokenized text
        for i in list_of_caption:
            j = word_tokenize(i)
            tok_text.append(j)
        bm25 = BM25Okapi(tok_text)
        tokenized_query = query.lower().split(" ")
        results = bm25.get_top_n(tokenized_query, tok_text, n=3)
        print(results)
        indexes=list()
        for i in results:
            if i in tok_text:
                indexes.append(tok_text.index(i))
        
        list_of_ids=list()
        for i in data["pics"]:
            list_of_ids.append(i["id"])
        
        retreived_images=list()

        for i in indexes:
            retreived_images.append(list_of_ids[i])
        print(retreived_images)
        return render_template('result_new.html',image1=retreived_images[0],image2=retreived_images[1],image3=retreived_images[2],query=query)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)