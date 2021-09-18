import cv2
import cvui
from imageai import Detection
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from flask import Flask, render_template, Response

app = Flask(__name__)

modelpath = "/home/k1ng/Desktop/yolo.h5"

rtsp_username = "admin"
rtsp_password = "admin12345"
width = 800
height = 480
cam_no = 7


yolo = Detection.ObjectDetection()
yolo.setModelTypeAsYOLOv3()
yolo.setModelPath(modelpath)
yolo.loadModel()


def create_camera(channel):
    rtsp = "rtsp://" + rtsp_username + ":" + rtsp_password + \
        "@192.168.1.100:554/Streaming/channels/" + \
            channel + "02"  # change the IP to suit yours
    cap = cv2.VideoCapture()
    cap.open(rtsp)
    cap.set(3, 640)  # ID number for width is 3
    cap.set(4, 480)  # ID number for height is 480
    cap.set(10, 100)  # ID number for brightness is 10qq
    return cap


def gen_frames():  # generate frame by frame from camera

    cam = create_camera(str(cam_no))

    while True:
        # Capture frame-by-frame

        success, frame = cam.read()  # read the camera frame

        if not success:
            break
        else:

             frame, preds = yolo.detectCustomObjectsFromImage(input_image=frame,
                      custom_objects=None, input_type="array",
                      output_type="array",
                      thread_safe=True,
                      minimum_percentage_probability=70,
                      display_percentage_probability=False,
                      display_object_name=True)

             ret, buffer = cv2.imencode('.jpg', frame)
             frame = buffer.tobytes()
             #yield (b'--frame\r\n'
                   #b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
             if len(preds)> 0:print (preds)


@app.route('/video_feed')
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0',threaded=False,debug=True)
