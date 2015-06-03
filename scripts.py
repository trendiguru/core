__author__ = 'Nadav Paz'

import urllib

import Utils
import background_removal


def dl_keyword_images(keyword):
    path = '/home/ubuntu/Dev/' + keyword
    for dress in keyword:
        dress_image = Utils.get_cv2_img_array(dress['image']['XLarge']['url'])
        if background_removal.image_is_relevant(background_removal.standard_resize(dress_image, 400)[0]):
            urllib.urlretrieve(dress['image']['XLarge']['url'], path + '/' + dress['id'])
