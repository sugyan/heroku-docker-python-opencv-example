from flask import Flask, make_response, render_template
from os import path
import cv2

app = Flask(__name__)
cascades_dir = path.normpath(path.join(cv2.__file__, '..', '..', '..', '..', 'share', 'OpenCV', 'haarcascades'))

@app.route('/image')
def image_with_facedetect():
    cascade = cv2.CascadeClassifier(path.join(cascades_dir, 'haarcascade_frontalface_alt.xml'))
    img = cv2.imread('./img/lena.jpg')
    rects = cascade.detectMultiScale(img, 1.3, 5)
    rects[:,2:] += rects[:,:2]
    x1, y1, x2, y2 = rects[0]
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    _, data = cv2.imencode('.png', img)
    resp = make_response(data.tobytes())
    resp.headers['Content-Type'] = 'image/png'
    return resp

@app.route('/')
def main():
    return render_template('index.html', version=cv2.__version__)
