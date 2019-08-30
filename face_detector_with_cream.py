import cv2
import os
import argparse
import pickle

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=False,
    help="path to trained model")
ap.add_argument("-l", "--le", type=str, required=False,
    help="path to label encoder", default="1e-4")
ap.add_argument("-d", "--detector", type=str, required=False,
    help="path to OpenCV's deep learning face detector", default="./face_detector")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args['detector'], "deploy.prototxt"])
# print(protoPath)
modelPath = os.path.sep.join([args["detector"],
    "res10_300x300_ssd_iter_140000.caffemodel"])
# print(modelPath)

net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)


# load the liveness detector model and label encoder from disk
print("[INFO] loading liveness detector...")

############################################

le = pickle.loads(open(args["le"], "rb").read())


# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream")