
import numpy as np
import caffe
import cv2
import skimage
from ..yonatan import yonatan_classifier
from .. import Utils

MODLE_FILE = "/home/yonatan/trendi/yonatan/resnet_50_gender_by_face/ResNet-50-deploy.prototxt"
PRETRAINED = "/home/yonatan/genderator_caffemodels/caffe_resnet50_snapshot_sgd_gender_by_face_iter_5000.caffemodel"

classifier = None


def load(gpu_device=None):
    if gpu_device is None:
        caffe.set_device(int(gpu_device))
    caffe.set_mode_gpu()
    image_dims = [224, 224]
    mean, input_scale = np.array([104.00699, 116.66877, 122.67892]), None
    channel_swap = [2, 1, 0]
    raw_scale = 255.0

    # Make classifier
    classifier = yonatan_classifier.Classifier(MODLE_FILE, PRETRAINED, image_dims=image_dims, mean=mean,
                                               input_scale=input_scale, raw_scale=raw_scale, channel_swap=channel_swap)

    print "Done initializing!"


def distance(v1, v2):
    raise NotImplementedError("Placeholder, not necessary for gender")


def execute(url_or_np_array, face_coordinates):

    print "Starting the genderism!"
    # check if i get a url (= string) or np.ndarray
    if isinstance(url_or_np_array, basestring):
        full_image = Utils.get_cv2_img_array(url_or_np_array)
    elif type(url_or_np_array) == np.ndarray:
        full_image = url_or_np_array
    else:
        return None

    #checks if the face coordinates are inside the image
    if full_image is None:
        print "not a good image"
        return None

    height, width, channels = full_image.shape

    x, y, w, h = face_coordinates

    if x > width or x + w > width or y > height or y + h > height:
        return None

    face_image = full_image[y: y + h, x: x + w]


    face_for_caffe = [skimage.img_as_float(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)).astype(np.float32)]

    if face_for_caffe is None:
        return None

    # Classify.
    predictions = classifier.predict(face_for_caffe)

    if predictions[0][1] > 0.7:
        return 'Male'
    else:
        return 'Female'
