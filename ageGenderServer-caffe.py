import os
os.environ['GLOG_minloglevel'] = '2'
import numpy as np
import sys
import cv2
import json

from flask import Response, Flask, send_file, make_response, request
from flask import render_template, jsonify
from io import BytesIO
from flask_cors import CORS, cross_origin
import urllib.request
import requests
import socket
import time
import caffe
sys.path.append("mtcnn")
import mtcnn

minsize = 40
threshold = [0.8, 0.8, 0.6]
factor = 0.709
caffe_model_path = "./mtcnn"
caffe.set_mode_cpu()
PNet = caffe.Net(caffe_model_path+"/det1.prototxt", caffe_model_path+"/det1.caffemodel", caffe.TEST)
RNet = caffe.Net(caffe_model_path+"/det2.prototxt", caffe_model_path+"/det2.caffemodel", caffe.TEST)
ONet = caffe.Net(caffe_model_path+"/det3.prototxt", caffe_model_path+"/det3.caffemodel", caffe.TEST)

imgSavePath = "static/uploadImages"
FACE_FEED_SIZE = 224    # age and gender input shape
models_dir="pretrained"
gender_model_def = models_dir+'/gender.prototxt'
gender_model_weights = models_dir+'/gender.caffemodel'
age_model_def = models_dir+'/age.prototxt'
age_model_weights =  models_dir+'/dex_imdb_wiki.caffemodel'
meanprotofile_path = models_dir+'/data_ilsvrc12_imagenet_mean.binaryproto'

def loadmean(meanprotopath):
    blob = caffe.proto.caffe_pb2.BlobProto()
    blob.ParseFromString(open(meanprotopath, 'rb').read())   
    return np.array(caffe.io.blobproto_to_array(blob))[0].mean(1).mean(1)

mean = loadmean(meanprotofile_path)

gender_net = caffe.Classifier(gender_model_def, gender_model_weights,
        image_dims=[224,224], mean=mean,raw_scale=255,
        channel_swap=[2,1,0])
age_net = caffe.Classifier(age_model_def, age_model_weights,
        image_dims=[224,224], mean=mean,raw_scale=255,
        channel_swap=[2,1,0])

def calcBiometry(img):

    img = cv2.resize(img, (FACE_FEED_SIZE, FACE_FEED_SIZE))[np.newaxis]
    age = age_net.predict(img)[0]
    gender = gender_net.predict(img)[0]
    return age, gender[0]

def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
 
    return ip

def process(img):

    boxPtArr = []
    ages = []
    genders = []
    boundingboxes, points = mtcnn.detect_face(img, minsize, PNet, RNet, ONet, threshold, False, factor)

    for i in range(boundingboxes.shape[0]):
        boxPtDict = {}
        box = []

        left = int(boundingboxes[i][0])
        right = int(boundingboxes[i][2])
        top = int(boundingboxes[i][1])
        bottom = int(boundingboxes[i][3])

        box.append(left)
        box.append(top)
        box.append(right)
        box.append(bottom)
        boxPtDict["box"] = box

        old_size = (right-left+bottom-top)/2.0
        centerX = right - (right-left)/2.0
        centerY = bottom - (bottom-top)/2 + old_size*0.1
        size = int(old_size*1.3)

        x1 = int(centerX-size/2)
        y1 = int(centerY-size/2)
        x2 = int(centerX+size/2)
        y2 = int(centerY+size/2)
        width = x2 - x1
        height = y2 - y1

        rectify_x1 = x1
        rectify_y1 = y1
        warped = img

        if(x2>img.shape[1]):
            warped = cv2.copyMakeBorder(img, 0, 0, 0, x2-img.shape[1], cv2.BORDER_CONSTANT)
        if(x1<0):
            warped = cv2.copyMakeBorder(img, 0, 0, -x1, 0, cv2.BORDER_CONSTANT)
            rectify_x1 = 0
        if(y2>img.shape[0]):
            warped = cv2.copyMakeBorder(img, 0, y2-img.shape[0], 0, 0, cv2.BORDER_CONSTANT)
        if(y1<0):
            warped = cv2.copyMakeBorder(img, -y1, 0, 0, 0, cv2.BORDER_CONSTANT)
            rectify_y1 = 0

        warped = warped[rectify_y1:y2, rectify_x1:x2]
        age, gender = calcBiometry(warped)
        boxPtDict["age"] = str(int(age.argmax()+0.5))
        if(gender>0.5):
            boxPtDict["gender"] = "Female"
        else:
            boxPtDict["gender"] = "Male"
        boxPtArr.append(boxPtDict)

    return boxPtArr


app = Flask(__name__)

@app.route("/ageGender", methods=['GET', 'POST'])
def ageGender():

    if request.method == 'POST':    # post by browser

        names = request.form['names']
        faceArr = []

        for name in names[:-1].split("*"):

            file = request.files[name]
            img = np.fromstring(file.read(), np.uint8)
            filename = imgSavePath +"/"+ str(time.time()) +".jpg"
            img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)
            cv2.imwrite(filename, img)
            boxPtArr = process(img)

            faceDict = {}
            faceDict["BoxsPoints"] = boxPtArr
            faceDict["name"] = file.filename
            faceArr.append(faceDict)

        faceJson = {}
        faceJson["faces"] = faceArr

        response = make_response(json.dumps(faceJson))

        response.headers['Access-Control-Allow-Origin'] = '*'
        return response

    else:

        # response = make_response(send_file(strIO, mimetype="image/jpeg"))
        response = make_response('{"faces":[{"name":"test1.jpg","box":[10,20,30,40],"pt":[10,20,30,40,50,10,20,30,40,50]},{"name":"test1.jpg","box":[101,20,30,40],"pt":[101,201,301,401,501,10,20,30,40,50]}]}')
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response


@app.route("/ageGenderBatch", methods=['POST'])
@cross_origin()
def ageGenderBatch():

    obj = request.get_json(force=True)
    urls = obj['urls']
    if(len(urls)>100):
        response = make_response("Limit Error: url items count > 100")
        return response

    faceArr = []
    for url in urls:
        try:
            headers = {"Referer":"https://www.fanfiction.net/s/4873155/1/Uchiha-Riz", "User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36"}

            content = requests.get(url, timeout=5, headers=headers).content

            img = np.fromstring(content, np.uint8)
            img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)
            filename = imgSavePath +"/"+ str(time.time()) +".jpg"
            with open(filename, "wb") as fp:
                fp.write(content)

            boxPtArr = process(img)

            faceDict = {}
            faceDict["BoxsPoints"] = boxPtArr
            faceDict["name"] = url
            faceArr.append(faceDict)

        except:
            response = make_response("Unfortunitely -- An Unknow Error Happened")
            return response

    faceJson = {}
    faceJson["faces"] = faceArr

    response = make_response(json.dumps(faceJson))
    return response

@app.route('/ageGender.html')
def home():
    ageGenderURL = "http://" +get_host_ip()+ ":" +str(port)+ "/ageGender"
    ageGenderBatchURL = "http://" +get_host_ip()+ ":" +str(port)+ "/ageGenderBatch"
    return render_template("ageGender.html", ageGender=ageGenderURL, ageGenderBatch=ageGenderBatchURL)


def main(argv):
    global port
    port = argv[1]

    app.run(host='0.0.0.0', port=port, debug=True)

if __name__ == '__main__':

    main(sys.argv)
