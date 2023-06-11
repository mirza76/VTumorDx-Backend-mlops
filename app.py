from flask import Flask, jsonify, request, Response, send_file, json
from flask_cors import CORS
import numpy as np
import cv2 as cv
from Models.VTumorDx_Models import VTumorDxModel
import os, io


app = Flask(__name__)
CORS(app)

model = VTumorDxModel()

CORS(app, origins=['http://localhost:4200'])

@app.route("/api/prediction", methods=["GET", "POST"])
def index():
    print("Processing ...")
    if request.method == "POST":
        image = request.files['image']
        image = image.read()
        img = np.frombuffer(image, np.uint8)
        img = cv.imdecode(img, cv.IMREAD_COLOR)
        img = cv.resize(img, (128, 128))

        tumor_type = model.predict_class(img)

        if tumor_type != 'no_tumor':
            img = model.tumor_segmentation(img)

        retImg, buffer = cv.imencode('.jpg', img)
        img_bytes = buffer.tobytes()
        
        response = Response(img_bytes, mimetype="image/jpeg")
        response.headers['tumor_type'] = tumor_type
        response.headers.add('Access-Control-Expose-Headers', 'tumor_type')

        return response
    
    return jsonify({
        "Message": "Method not supported"
    })

if __name__ == "__main__":
    app.run(debug=True)
