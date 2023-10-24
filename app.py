import os
import io
import numpy as np
import cv2
from waitress import serve
from controllers.analyze import analyze
from werkzeug.utils import secure_filename
from flask import \
        Flask, \
        render_template, \
        request, \
        abort, \
        redirect, \
        url_for


app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000
app.config['UPLOAD_FOLDER'] = './static/tmp'
ALLOWED_EXTENTIONS = set(['png', 'jpg', 'PNG', 'JPG'])
IMG_WIDTH = 500

is_model_img_set = False
is_img_set = False
img_bin = ""
img_name = ""
model_img_np = None


@app.route('/')
def index():
    return render_template('index.html', title="GS_app")


@app.route('/result')
def result():
    if is_img_set:
        print("is_img_set: True")
        return render_template(
               'result.html',
               title="Result",
               img_name=img_name)
    elif not is_img_set:
        print("is_img_set: False")
        abort(404)
        return render_template('404.html'), 404


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    global is_model_img_set
    if not is_model_img_set:
        return redirect(url_for('upload_model'))

    if request.method == 'POST':
        # Receive image file
        file = request.files['file']

        # Prevent traversal attack => secure_filename
        fileName = secure_filename(file.filename)

        file = file.stream.read()
        file = io.BytesIO(file)
        file = np.asarray(bytearray(file.read()), dtype=np.uint8)
        cv2_img = cv2.imdecode(file, cv2.IMREAD_COLOR)
        cv2_img = cv2.resize(cv2_img,
                             (IMG_WIDTH,
                              int(IMG_WIDTH*cv2_img.shape[0] /
                                  cv2_img.shape[1])))

        global model_img_np

        img = analyze(cv2_img, model_img_np)

        img_path = os.path.join(app.config['UPLOAD_FOLDER'], fileName)
        cv2.imwrite(img_path, img)

        global is_img_set
        global img_name
        is_img_set = True
        img_name = fileName

        return redirect(url_for('result'))
    else:
        return render_template('upload.html', title="upload")


@app.route('/upload_model', methods=['GET', 'POST'])
def upload_model():
    if request.method == 'POST':
        file = request.files['file']

        global model_img_np
        global is_model_img_set

        file = file.stream.read()
        file = io.BytesIO(file)
        file = np.asarray(bytearray(file.read()), dtype=np.uint8)
        model_img_np = cv2.imdecode(file, cv2.IMREAD_COLOR)
        model_img_np = cv2.resize(model_img_np,
                                  (IMG_WIDTH,
                                   int(IMG_WIDTH*model_img_np.shape[0] /
                                       model_img_np.shape[1])))

        is_model_img_set = True

        return redirect(url_for('upload_file'))
    else:
        return render_template('setting.html', title="setting")


@app.route('/setting')
def setting():
    return render_template('setting.html', title="setting")


if __name__ == "__main__":
    serve(app, host='0.0.0.0', port=5000, threads=10)
