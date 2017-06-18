__author__ = 'Nadav Paz'
import os
import cv2

from trendi import background_removal

def compare(img_arr):
    ffc = background_removal.find_face_ccv(img_arr, max_num_of_faces=100)
    print('ffc:'+str(ffc))

    ff_cascade = background_removal.find_face_cascade(img_arr, max_num_of_faces=10)
    print('ffcas:'+str(ff_cascade))

    ff_dlib = background_removal.find_face_dlib_with_scores(img_arr, max_num_of_faces=100)
    print('ffdlib:'+str(ff_dlib))


if __name__ == "__main__":
    dir = '/media/jeremy/9FBD-1B00/data/jeremy/image_dbs/mongo'
    subdirs = [os.path.join(dir, f) for f in os.listdir(dir)]
    for subdir in subdirs:
        files = [os.path.join(subdir,f) for f in os.listdir(subdir)]
        print('{} files in {}'.format(len(files),subdir))
        for file in files :
            img_arr = cv2.imread(file)
            if img_arr is None:
                print('got none img arr for {}'.format(file))
                continue
            compare(img_arr)

    # background_removal.choose_faces(img_arr, faces_list, max_num_of_faces)
    # background_removal.score_face(face, image)
    # background_removal.face_is_relevant(image, face)
    # background_removal.is_skin_color(face_ycrcb):
# variance_of_laplacian(image):
# is_one_color_image(image):
# average_bbs(bb1, bb2):
# combine_overlapping_rectangles(bb_list):
# body_estimation(image, face):
# bb_mask(image, bounding_box):
# paperdoll_item_mask(item_mask, bb):
# create_mask_for_gc(rectangles, image):
# create_arbitrary(image):
#
# face_skin_color_estimation(image, face_rect):
# person_isolation(image, face):
# check_skin_percentage(image):
