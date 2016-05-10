__author__ = 'jeremy'
import os
import cv2

from trendi import pipeline
from trendi import imutils


def show_pd_results(dir):
    ########WARNING NOT FINISHED
    filenames = [os.path.join]
    for f in filenames:
    print('sending img '+f)
    im = cv2.imread(f)
    mask, labels = retval.result[:2]
    parse_name = f.split('.jpg')[0]+'_parse.png'
    cv2.imwrite(parse_name,mask)
    print('labels:'+str(labels))
    sorted_labels=sorted(labels.items(),key=operator.itemgetter(1))
    print('sorted labels :'+str(sorted_labels))
    labs_only = [i[0] for i in sorted_labels]
    print('labsonly '+str(labs_only))
    imutils.show_mask_with_labels(parse_name,labs_only,save_images=True)

    aftermask = pipeline.after_pd_conclusions(mask, labels, face=None)
    after_pd_conclusions_name = parse_name.split('_parse.png')[0]+'_after_pd_conclusions.png'
    cv2.imwrite(after_pd_conclusions_name,aftermask)
    imutils.show_mask_with_labels(after_pd_conclusions_name,labs_only,save_images=True)

    if retval is not None:
        print('retval:' + str(retval.result)+' time:'+str(dt))
    else:
        print('no return val (None)')


if __name__ =="__main__":
    files = [os.path.join(dir,f) for f in dir if '.png']

    pipeline.after_nn_conclusions(mask, labels)