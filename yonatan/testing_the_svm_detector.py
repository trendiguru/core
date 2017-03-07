#!/usr/bin/env python

import pymongo
from trendi.yonatan import test_object_detector, functions


# if i run this function on brainik80a:
# db = pymongo.MongoClient().mydb

# if in_docker:
db = pymongo.MongoClient('localhost', port=27017).mydb

# else:
# db = constants.db

# irrelevant_dictionary = db.irrelevant_images_distinct.find()
irrelevant_dictionary = list(db.irrelevant_images_distinct.aggregate([{ "$sample": { "size": 5000} }]))

results_text_file = open("/data/yonatan/yonatan_files/trendi/yonatan/txt_files/results_irrelevant_db_C_40_symmetry.txt", "w")

error_counter = 0
face_image_counter = 0
found_dress_counter = 0
no_dress_counter = 0


for i in range(1, len(irrelevant_dictionary)):

    # if i % 10 != 0:
    #     continue

    link_to_image = irrelevant_dictionary[i]['image_urls'][0]

    full_image = functions.url_to_np_array(link_to_image)

    if full_image is None:
        error_counter += 1
        continue

    # check if there's faces in the image
    is_there_faces = functions.find_face_dlib(full_image)

    if is_there_faces["are_faces"]:
        face_image_counter += 1
        continue

    if test_object_detector.detect(full_image, found_dress_counter):
        found_dress_counter += 1
    else:
        no_dress_counter += 1

    total = error_counter + face_image_counter + found_dress_counter + no_dress_counter
    added_images = found_dress_counter / float(total)
    print "error_counter = {0}, face_image_counter = {1}, found_dress_counter = {2}, no_dress_counter = {3}\n" \
          "total = {4}, added_images_percent = {5}".format(error_counter, face_image_counter, found_dress_counter,
                                                           no_dress_counter, total, added_images)

    results_text_file.write("svm_045 - C = 40 - symmetry, svm_07 - C = 40 - symmetry\nerror_counter = " + str(error_counter) + ", face_image_counter = " +
                            str(face_image_counter) + ", found_dress_counter = " + str(found_dress_counter) +
                            ", no_dress_counter = " + str(no_dress_counter) + ", total = " + str(total) +
                            ", added_images_percent = " + str(added_images) + "\n")


results_text_file.flush()
