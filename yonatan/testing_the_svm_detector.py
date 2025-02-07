#!/usr/bin/env python

from trendi.yonatan import test_object_detector, functions


db = functions.connect_to_mongo()

# irrelevant_dictionary = db.irrelevant_images_distinct.find()
irrelevant_dictionary = list(db.irrelevant_images_distinct.aggregate([{ "$sample": { "size": 5000} }]))

results_text_file = open("/data/yonatan/yonatan_files/trendi/yonatan/txt_files/results_irrelevant_db_C_40_symmetry_take2.txt", "w")

error_counter = 0
face_image_counter = 0
found_dress_counter = 0
no_dress_counter = 0
unique_counter = 0

for i in range(1, len(irrelevant_dictionary)):

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

    is_there_dress = test_object_detector.detect(full_image, found_dress_counter)

    if is_there_dress:
        found_dress_counter += 1
        unique_counter += is_there_dress - 1
    else:
        no_dress_counter += 1

    total = error_counter + face_image_counter + found_dress_counter + no_dress_counter
    added_images = found_dress_counter / float(total)
    print "error_counter = {0}, face_image_counter = {1}, found_dress_counter = {2}, no_dress_counter = {3}, unique_counter = {4}\n" \
          "total = {5}, added_images_percent = {6}".format(error_counter, face_image_counter, found_dress_counter,
                                                           no_dress_counter, unique_counter, total, added_images)

    results_text_file.write("svm_045 - C = 40 - symmetry, svm_07 - C = 40 - symmetry\nerror_counter = " + str(error_counter) + ", face_image_counter = " +
                            str(face_image_counter) + ", found_dress_counter = " + str(found_dress_counter) +
                            ", no_dress_counter = " + str(no_dress_counter) + ", unique_counter = " + str(unique_counter) + ", total = " + str(total) +
                            ", added_images_percent = " + str(added_images) + "\n")


results_text_file.flush()
