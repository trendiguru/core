import cv2
import os
import paperdoll_parse_enqueue

def show_pd_results(file_base_name):
    bmp=file_base_name+'.bmp'
    jpg=file_base_name+'.jpg'
    pose=file_base_name+'.pose'

    print('reading '+str(jpg))
    img_arr = cv2.imread(jpg)
    if img_arr is not None:
        cv2.imshow('orig',img_arr)
    else:
        print('coundlt get jpg at '+str(jpg))

    paperdoll_parse_enqueue.colorbars()

    print('reading '+str(bmp))
    img_arr = cv2.imread(bmp)
    if img_arr is not None:
        paperdoll_parse_enqueue.show_parse(img_array = img_arr)
    else:
        print('coundlt get png at '+str(bmp))


if __name__ == '__main__':
    path = '/home/jr/tg/pd_output'
    files = ['56558cd462532224c676ba7c']
    for file in files:
        fullpath = os.path.join(path,file)
        show_pd_results(fullpath)
        raw_input('enter for next')