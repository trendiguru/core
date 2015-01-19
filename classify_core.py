#!/usr/bin/env python
__author__ = 'liorsabag'

import cv2
import Utils
import logging


def classify_image_with_classifiers(url_or_path_to_image_file_or_cv2_image_array,
                                    *classifier_xml_list, **classifier_xml_dict):
    """

    :param url_or_path_to_image_file_or_cv2_image_array: image to be classified
    :param classifier_xml_list: list of paths to classifier xml files
    :param classifier_xml_dict: keys are classifier names and values are corresponding xml files
    :return: dictionary of {key (if provided)/classifier_xml : [list of found bounding_boxes]}
    """
    if classifier_xml_list:
        classifier_xml_dict = {xml: xml for xml in classifier_xml_list}

    cascades_dict = {}

    for classifier_key, classifier_xml in classifier_xml_dict.iteritems():
        classifier = cv2.CascadeClassifier(classifier_xml)
        if not classifier.empty():
            cascades_dict[classifier_key] = classifier
        else:
            logging.warning("Could not load " + classifier_xml)

    img = Utils.get_cv2_img_array(url_or_path_to_image_file_or_cv2_image_array)

    bounding_box_dict = {key: classifier.detectMultiScale(img) for key, classifier in cascades_dict.iteritems()}

    return bounding_box_dict

