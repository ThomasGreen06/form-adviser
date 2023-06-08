import os
import io
import numpy as np
from flask import Flask, render_template, request, abort, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename
import cv2, uuid, base64, re

from controllers.analyze import analyze

app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000
app.config['UPLOAD_FOLDER'] = './static/tmp'
ALLOWED_EXTENTIONS = set(['png', 'jpg', 'PNG', 'JPG'])
IMG_WIDTH = 500

is_img_set = False
img_bin = ""
img_name = ""

@app.route('/')
def index():
    return render_template('index.html', title="GS_app")


@app.route('/result')
def result():
    if is_img_set:
        print("is_img_set: True")
        return render_template('result.html', title="Result", img_name=img_name)
    elif not is_img_set:
        print("is_img_set: False")
        abort(404)
        return render_template('404.html'), 404
    


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Receive image file
        file = request.files['file']

        # Prevent traversal attack => secure_filename
        fileName = secure_filename(file.filename)

        content_type = ""
        if 'png'in file.content_type:
            content_type = 'png'
        elif 'jpg' in file.content_type:
            content_type = 'jpeg'

        # Convert to cv2-image
        file = file.stream.read()
        file = io.BytesIO(file)
        file = np.asarray(bytearray(file.read()), dtype=np.uint8)
        cv2_img = cv2.imdecode(file, cv2.IMREAD_COLOR)
        cv2_img = cv2.resize(cv2_img, (IMG_WIDTH, int(IMG_WIDTH*cv2_img.shape[0]/cv2_img.shape[1])))

        #file.save(os.path.join(app.config['UPLOAD_FOLDER'], fileName))

        img = analyze(cv2_img)

        img_path = os.path.join(app.config['UPLOAD_FOLDER'], fileName)
        cv2.imwrite(img_path, img)


        img_base64 = base64.b64encode(img).decode("ascii")
        img_base64 = f'data:image/jpeg;base64,{img_base64}'
        #img_base64 = img_base64.format(img_base64)

        global is_img_set
        global img_name
        is_img_set = True
        img_name = fileName

        return redirect(url_for('result'))
    else:
        return render_template('upload.html', title="upload")


@app.route('/setting')
def setting():
    return render_template('setting.html', title="setting")


if __name__ == "__main__":
    app.run(debug=True)

