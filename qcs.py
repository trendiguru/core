__author__ = 'Nadav Paz'

import logging
import os
import binascii

import pymongo
import cv2
import redis
from rq import Queue

import boto3
import background_removal
import Utils


db = pymongo.MongoClient().mydb
images = pymongo.MongoClient().mydb.images
r = redis.Redis()
q2 = Queue('send_to_categorize', connection=r)
q3 = Queue('receive_categories', connection=r)
q4 = Queue('send_to_bb', connection=r)
q5 = Queue('receive_bb', connection=r)
q6 = Queue('send_20s_results', connection=r)
q7 = Queue('receive_20s_results', connection=r)
q8 = Queue('send_last_20', connection=r)
q9 = Queue('receive_final_results', connection=r)


def upload_image(image, name, bucket_name=None):
    image_string = cv2.imencode(".jpg", image)[1].tostring()
    bucket_name = bucket_name or "tg-boxed-faces"
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(name=bucket_name)
    bucket.put_object(Key="{0}.jpg".format(name), Body=image_string, ACL='public-read', ContentType="image/jpg")
    return "{0}/{1}/{2}.jpg".format("https://s3.eu-central-1.amazonaws.com", bucket_name, name)


# q1 - images queue

# FUNCTION 1
def from_image_url_to_task1(image_url):
    image_obj = images.find_one({"image_url": image_url})
    if not image_obj:  # new image
        image = background_removal.standard_resize(Utils.get_cv2_img_array(image_url), 400)[0]
        if image is None:
            logging.warning("There's no image in the url!")
            return None
        relevance = background_removal.image_is_relevant(image)
        image_dict = {'image_url': image_url, 'relevant': relevance.is_relevant}
        if relevance.is_relevant:
            image_dict['people'] = []
            for face in relevance.faces:
                x, y, w, h = face
                person = {'face': face.tolist(), 'person_id': binascii.hexlify(os.urandom(32))}
                copy = image.copy()
                cv2.rectangle(copy, (x, y), (x + w, y + h), [0, 255, 0], 2)
                image_s3_url = upload_image(copy, person['person_id'])
                person['url'] = image_s3_url
                image_dict['people'].append(person)
                # q2.enqueue(send_image_to_qc_categorization, image_s3_url, image_dict)
        else:
            logging.warning('image is not relevant, but stored anyway..')
        image_obj_id = images.insert(image_dict)
        image_obj = images.find_one({'image_url': image_url})
        return image_obj
    else:
        # understand which details are already strored and react accordingly
        return image_obj
        # END OF FUNCTION 1


# q2

# FUNCTION 2
# def send_image_to_qc_categorization(copy):

# END OF FUNCTION 2
"""
            # q3

            # FUNCTION 3
            items = get_categorization_from_qc()
            determine_final_categories(items)
            for item in items:
                # END OF FUNCTION 3

                # q4

                # FUNCTION 4
                send_bb_task_to_qc(copy, item.category)
                # END

                # q5

                # FUNCTION 5
                bb_list = get_bb_list_from_qc()
                bb = determine_final_bb(bb_list)
                fp, results, svg = find_similar_mongo.got_bb(image_url, image_id, item_id, bb, 100, item.category)
                # END

                # q6

                # FUNCTION 6
                send_100_results_to_qc_in_20s(copy, results)
                # END

                # q7

                # FUNCTION 7
                sorted_results = get_sorted_results_from_qc()
                final_20_results = rearrange_results(sorted_results)
                # END

                # q8

                # FUNCTION 8
                send_final_20_results_to_qc_in_10s(copy, final_20_results)
                # END

                # q9

                # FUNCTION 9
                final_results = get_final_results_from_qc()
                insert_final_results(item.id, final_results)
"""
