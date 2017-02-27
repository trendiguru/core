#!/usr/bin/python
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
# This example program shows how you can use dlib to make an object
#   detector for things like faces, pedestrians, and any other semi-rigid
#   object.  In particular, we go though the steps to train the kind of sliding
#   window object detector first published by Dalal and Triggs in 2005 in the
#   paper Histograms of Oriented Gradients for Human Detection.
#
#
# COMPILING/INSTALLING THE DLIB PYTHON INTERFACE
#   You can install dlib using the command:
#       pip install dlib
#
#   Alternatively, if you want to compile dlib yourself then go into the dlib
#   root folder and run:
#       python setup.py install
#   or
#       python setup.py install --yes USE_AVX_INSTRUCTIONS
#   if you have a CPU that supports AVX instructions, since this makes some
#   things run faster.
#
#   Compiling dlib should work on any operating system so long as you have
#   CMake and boost-python installed.  On Ubuntu, this can be done easily by
#   running the command:
#       sudo apt-get install libboost-python-dev cmake
#
#   Also note that this example requires scikit-image which can be installed
#   via the command:
#       pip install scikit-image
#   Or downloaded from http://scikit-image.org/download.html.

import os
import sys
import glob
import numpy as np
from trendi.yonatan import preparing_data_from_db, grabCut

import dlib
from skimage import io
import cv2


images_new2 = []
boxes_new2 = []

images_new_test2 = []
boxes_new_test2 = []


images_new = []
boxes_new = []

images_new_test = []
boxes_new_test = []

# In this example we are going to train a face detector based on the small
# faces dataset in the examples/faces directory.  This means you need to supply
# the path to this faces folder as a command line argument so we will know
# where it is.

# if len(sys.argv) != 2:
#     print(
#         "Give the path to the examples/faces directory as the argument to this "
#         "program. For example, if you are in the python_examples folder then "
#         "execute this program by running:\n"
#         "    ./train_object_detector.py ../examples/faces")
#     exit()
# faces_folder = sys.argv[1]


# Now let's do the training.  The train_simple_object_detector() function has a
# bunch of options, all of which come with reasonable default values.  The next
# few lines goes over some of these options.
options = dlib.simple_object_detector_training_options()
# Since faces are left/right symmetric we can tell the trainer to train a
# symmetric detector.  This helps it get the most value out of the training
# data.
options.add_left_right_image_flips = False
# The trainer is a kind of support vector machine and therefore has the usual
# SVM C parameter.  In general, a bigger C encourages it to fit the training
# data better but might lead to overfitting.  You must find the best C value
# empirically by checking how well the trained detector works on a test set of
# images you haven't trained on.  Don't just leave the value set at 5.  Try a
# few different C values and see what works best for your data.
options.C = 20
# Tell the code how many CPU cores your computer has for the fastest training.
options.num_threads = 12
options.be_verbose = True
options.epsilon = 0.001


# training_xml_path = os.path.join(faces_folder, "training.xml")
# testing_xml_path = os.path.join(faces_folder, "testing.xml")
# # This function does the actual training.  It will save the final detector to
# # detector.svm.  The input is an XML file that lists the images in the training
# # dataset and also contains the positions of the face boxes.  To create your
# # own XML files you can use the imglab tool which can be found in the
# # tools/imglab folder.  It is a simple graphical tool for labeling objects in
# # images with boxes.  To see how to use it read the tools/imglab/README.txt
# # file.  But for this example, we just use the training.xml file included with
# # dlib.
# dlib.train_simple_object_detector(training_xml_path, "detector.svm", options)


## train set ##
counter_bad = 0
counter = 0
counter2 = 0
counter_big_ratio = 0
counter_big_ratio2 = 0

number_of_images = 100

sum_w = 0
sum_h = 0

# when i limited h_cropped and w_cropped to be bigger than (20, 20) and
# also when (50, 100) : (w , h), the average w and h were:
average_w = 150
average_h = 331
w_h_ratio = float(average_w) / average_h  # = 0.453

w_h_ratio2 = 0.7

break_from_main_loop = False

for root, dirs, files in os.walk('/data/dress_detector/images_raw'):
    if not break_from_main_loop:
        for file in files:
            ## if i want to limit to smaller number of images
            if counter > number_of_images:
                print "counter: {0}, counter_bad : {1}, counter_big_ratio : {2}\ncounter2 : {3}, counter_big_ratio2: {4}".format(counter, counter_bad, counter_big_ratio, counter2, counter_big_ratio2)
                break

            full_image = cv2.imread('/data/dress_detector/images_raw/' + file)

            original_image = full_image.copy()
            h_original, w_original, d_original = original_image.shape

            # # if there's a head, cut it off
            faces = preparing_data_from_db.find_face_dlib(full_image)

            x_face, y_face, w_face, h_face = 0, 0, 0, 0  # default them to 0 - in case there's no face in the image
            if faces["are_faces"]:
                if len(faces['faces']) == 1:
                    x_face, y_face, w_face, h_face = faces['faces'][0]
                    full_image = full_image[y_face + h_face:, :, :]  # Crop the face from the image
                    # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
                else:
                    print "more than one face"
                    counter_bad += 1
                    continue

            # h_gap = the gap between (y_face + h_face) to y_cropped (most high y value of cropped image)
            # w_gap = the gap between 0 (x_original) to x_cropped (most left x value of cropped image)
            try:
                cropped_image, h_gap, w_gap = grabCut.grabcut(full_image)
            except:
                counter_bad += 1
                continue

            if cropped_image is None:
                counter_bad += 1
                continue

            h_cropped, w_cropped, d_cropped = cropped_image.shape

            if w_cropped < 50 or h_cropped < 100:
                print "BB too small"
                counter_bad += 1
                continue

            # w going to stay the same, h going to be different - i'm going to chop it from bottom
            # ratio = w / h -> h = w / ratio
            new_h_cropped = int(w_cropped / w_h_ratio)

            h_cropped_out_of_bound = False
            if y_face + h_face + h_gap + new_h_cropped > h_original:
                # new_h_cropped = h_original - (y_face + h_face + h_gap + 1)
                temp_new_h = int(w_cropped / w_h_ratio2)
                new_h_cropped = temp_new_h + y_face + h_face + h_gap
                h_cropped_out_of_bound = True
                counter_big_ratio += 1
                if new_h_cropped > h_original:
                    print "ratio too big even for ratio2"
                    counter_big_ratio2 += 1
                    continue
            else:
                new_h_cropped += y_face + h_face + h_gap

            new_w_h_ratio = float(w_gap + w_cropped - w_gap) / (new_h_cropped - (y_face + h_face + h_gap))
            if new_w_h_ratio > 0.5:
                print "still ratio bigger than 0.5!"
                counter_big_ratio += 1
                # continue

            # line_in_list_boxes = ([dlib.rectangle(left=w_gap, top=y_face + h_face + h_gap, right=w_cropped, bottom=new_h_cropped)])
            line_in_list_boxes = [dlib.rectangle(left=w_gap, top=y_face + h_face + h_gap, right=w_gap + w_cropped, bottom=new_h_cropped)]
            print "left=w_gap = {0}, top=y_face + h_face + h_gap = {1}, right=w_gap + w_cropped = {2}, bottom=new_h_cropped = {3}".format(w_gap, y_face + h_face + h_gap, w_gap + w_cropped, new_h_cropped)
            print "width = {0}, height = {1}".format(w_gap + w_cropped - w_gap, new_h_cropped - (y_face + h_face + h_gap))
            print "ratio_w_h = {0}, h_cropped_out_of_bound = {1}".format(new_w_h_ratio, h_cropped_out_of_bound)


            try:
                # line_in_list_images = cv2.imread('/data/dress_detector/images_raw/' + file)
                line_in_list_images = io.imread('/data/dress_detector/images_raw/' + file)
            except:
                print "bad image!!"
                counter_bad += 1
                continue

            if h_cropped_out_of_bound:
                counter2 += 1
                if counter + counter_bad <= number_of_images * 0.8:
                    boxes_new2.append(line_in_list_boxes)
                    images_new2.append(line_in_list_images)
                elif number_of_images * 0.8 < counter + counter_bad < number_of_images:
                    boxes_new_test2.append(line_in_list_boxes)
                    images_new_test2.append(line_in_list_images)
                else:
                    print "counter: {0}, counter_bad : {1}, counter_big_ratio : {2}\ncounter2 : {3}, counter_big_ratio2: {4}".format(
                        counter, counter_bad, counter_big_ratio, counter2, counter_big_ratio2)
                    break_from_main_loop = True
                    break
            else:
                counter += 1
                if counter + counter_bad <= number_of_images * 0.8:
                    boxes_new.append(line_in_list_boxes)
                    images_new.append(line_in_list_images)
                elif number_of_images * 0.8 < counter + counter_bad < number_of_images:
                    boxes_new_test.append(line_in_list_boxes)
                    images_new_test.append(line_in_list_images)
                else:
                    print "counter: {0}, counter_bad : {1}, counter_big_ratio : {2}\ncounter2 : {3}, counter_big_ratio2: {4}".format(
                        counter, counter_bad, counter_big_ratio, counter2, counter_big_ratio2)
                    break_from_main_loop = True
                    break

            print "counter: {}".format(counter)

#             sum_w += w_cropped
#             sum_h += h_cropped
#
#
#
#
#
#
#
# average_w = sum_w / (counter + 1)
# average_h = sum_h / (counter + 1)
#
# print "average_w: {0}\naverage_h: {1}".format(average_w, average_h)



# images_array = np.load('/data/dress_detector/images_small_set_save.npy')
# boxes_array = np.load('/data/dress_detector/boxes_small_set_save.npy')
#
# images = images_array.tolist()
# boxes = boxes_array.tolist()

detector = dlib.train_simple_object_detector(images_new, boxes_new, options)
print "Done training!"
detector.save('/data/detector_0.45.svm')
print "Done saving!"

detector2 = dlib.train_simple_object_detector(images_new2, boxes_new2, options)
print "Done training!"
detector2.save('/data/detector_0.7.svm')
print "Done saving!"

# # We can look at the HOG filter we learned.  It should look like a face.  Neat!
# win_det = dlib.image_window()
# win_det.set_image(detector2)
#
# # Now let's look at its HOG filter!
# # win_det.set_image(detector2)
# dlib.hit_enter_to_continue()
#
# # Note that you don't have to use the XML based input to
# # test_simple_object_detector().  If you have already loaded your training
# # images and bounding boxes for the objects then you can call it as shown
# # below.
# print("\nTraining accuracy: {}".format(
#     dlib.test_simple_object_detector(images, boxes, detector2)))
#
#
# # # Now that we have a face detector we can test it.  The first statement tests
# # # it on the training data.  It will print(the precision, recall, and then)
# # # average precision.
# # print("")  # Print blank line to create gap from previous output
# # print("Training accuracy: {}".format(
# #     dlib.test_simple_object_detector(training_xml_path, "detector.svm")))
# # # However, to get an idea if it really worked without overfitting we need to
# # # run it on images it wasn't trained on.  The next line does this.  Happily, we
# # # see that the object detector works perfectly on the testing images.
# # print("Testing accuracy: {}".format(
# #     dlib.test_simple_object_detector(testing_xml_path, "detector.svm")))
#
#
#
#
#
# # # Now let's use the detector as you would in a normal application.  First we
# # # will load it from disk.
# # detector = dlib.simple_object_detector("detector.svm")
#
# # We can look at the HOG filter we learned.  It should look like a face.  Neat!
# win_det = dlib.image_window()
# win_det.set_image(detector)
#
# # Now let's run the detector over the images in the faces folder and display the
# # results.
# print("Showing detections on the images in the faces folder...")
# win = dlib.image_window()
# for f in glob.glob(os.path.join(faces_folder, "*.jpg")):
#     print("Processing file: {}".format(f))
#     img = io.imread(f)
#     dets = detector(img)
#     print("Number of faces detected: {}".format(len(dets)))
#     for k, d in enumerate(dets):
#         print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
#             k, d.left(), d.top(), d.right(), d.bottom()))
#
#     win.clear_overlay()
#     win.set_image(img)
#     win.add_overlay(dets)
#     dlib.hit_enter_to_continue()







# # Finally, note that you don't have to use the XML based input to
# # train_simple_object_detector().  If you have already loaded your training
# # images and bounding boxes for the objects then you can call it as shown
# # below.
#
# # You just need to put your images into a list.
# images = [io.imread(faces_folder + '/2008_002506.jpg'),
#           io.imread(faces_folder + '/2009_004587.jpg')]
# # Then for each image you make a list of rectangles which give the pixel
# # locations of the edges of the boxes.
# boxes_img1 = ([dlib.rectangle(left=329, top=78, right=437, bottom=186),
#                dlib.rectangle(left=224, top=95, right=314, bottom=185),
#                dlib.rectangle(left=125, top=65, right=214, bottom=155)])
# boxes_img2 = ([dlib.rectangle(left=154, top=46, right=228, bottom=121),
#                dlib.rectangle(left=266, top=280, right=328, bottom=342)])
# # And then you aggregate those lists of boxes into one big list and then call
# # train_simple_object_detector().
# boxes = [boxes_img1, boxes_img2]
#
# images_array = np.load(open('/data/dress_detector/images.npy', 'rb'))
# boxes_array = np.load(open('/data/dress_detector/boxes.npy', 'rb'))
#
# images = images_array.tolist()
# boxes = boxes_array.tolist()
#
# detector2 = dlib.train_simple_object_detector(images, boxes, options)
# # We could save this detector to disk by uncommenting the following.
# detector2.save('detector2.svm')
#
# # Now let's look at its HOG filter!
# win_det.set_image(detector2)
# dlib.hit_enter_to_continue()
#
# # Note that you don't have to use the XML based input to
# # test_simple_object_detector().  If you have already loaded your training
# # images and bounding boxes for the objects then you can call it as shown
# # below.
print "detector_0.45:"
print("\nTraining accuracy: {}".format(
    dlib.test_simple_object_detector(images_new, boxes_new, detector)))

print("\nTesting accuracy: {}".format(
    dlib.test_simple_object_detector(images_new_test, boxes_new_test, detector)))


print "detector_0.7:"
print("\nTraining accuracy: {}".format(
    dlib.test_simple_object_detector(images_new2, boxes_new2, detector2)))

print("\nTesting accuracy: {}".format(
    dlib.test_simple_object_detector(images_new_test2, boxes_new_test2, detector2)))
