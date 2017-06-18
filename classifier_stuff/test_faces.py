__author__ = 'Nadav Paz'
import os
import cv2

from trendi import background_removal

def test_skin_from_face(img_arr)
    ff_cascade = background_removal.find_face_cascade(img_arr, max_num_of_faces=10)
    print('ffcas:'+str(ff_cascade))
    if ff_cascade['are_faces'] :
        faces = ff_cascade['faces']
        if faces == []:
            print('ffascade reported faces but gave none')
        else:
            img_ff_cascade = img_arr.copy()

            for face in faces:
                print('cascade face:{}'.format(face))
                cv2.rectangle(img_ff_cascade,(face[0],face[1]),(face[0]+face[2],face[1]+face[3]),(255,100,0),5)
                fsc = background_removal.face_skin_color_estimation()
                print('face skin color {}'.format(fsc))
            cv2.imshow('ffcascade',img_ff_cascade)


def compare_detection_methods(img_arr):
    cv2.imshow('orig',img_arr)

    ffc = background_removal.find_face_ccv(img_arr, max_num_of_faces=100)
    print('ccv:'+str(ffc))
    if ffc['are_faces'] :
        faces = ffc['faces']
        img_ffc = img_arr.copy()
        for face in faces:
            print('ccv face:{}'.format(face))
            cv2.rectangle(img_ffc,(face[0],face[1]),(face[0]+face[2],face[1]+face[3]),(255,100,0),5)
        cv2.imshow('ffc',img_ffc)

    ff_cascade = background_removal.find_face_cascade(img_arr, max_num_of_faces=10)
    print('ffcas:'+str(ff_cascade))
    if ff_cascade['are_faces'] :
        faces = ff_cascade['faces']
        if faces == []:
            print('ffascade reported faces but gave none')
        else:
            img_ff_cascade = img_arr.copy()

            for face in faces:
                print('cascade face:{}'.format(face))
                cv2.rectangle(img_ff_cascade,(face[0],face[1]),(face[0]+face[2],face[1]+face[3]),(255,100,0),5)
            cv2.imshow('ffcascade',img_ff_cascade)


    ff_dlib = background_removal.find_face_dlib(img_arr, max_num_of_faces=100)
    print('ffdlib:'+str(ff_dlib))
    if ff_dlib['are_faces'] :
        faces = ff_dlib['faces']
        img_ff_dlib = img_arr.copy()
        for face in faces:
            print('dlib face:{}'.format(face))
            cv2.rectangle(img_ff_dlib,(face[0],face[1]),(face[0]+face[2],face[1]+face[3]),(255,100,0),5)
        cv2.imshow('ff_dlib',img_ff_dlib)
    cv2.waitKey(0)

def check_dir(dir):
    print('doing dir {}'.format(dir))
    files = [os.path.join(dir,f) for f in os.listdir(dir) if not os.path.isdir(os.path.join(dir,f))]
    print('{} files in {}'.format(len(files),dir))
    for f in files :
        img_arr = cv2.imread(f)
        if img_arr is None:
            print('got none img arr for {}'.format(f))
            continue
        compare_detection_methods(img_arr)

if __name__ == "__main__":
    dir = '/media/jeremy/9FBD-1B00/data/jeremy/image_dbs/mongo/amazon_us_female/suit/'
    check_dir(dir)
    # subdirs = [os.path.join(dir, d) for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    # print('{} subdirs in {}'.format(len(subdirs),dir))
    # for subdir in subdirs:
    #     check_dir(subdir)



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
