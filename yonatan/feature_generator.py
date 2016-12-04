#!/usr/bin/env python

from ..yonatan import edit_names, cropping, resize_and_save, preparing_txt_files, yonatan_constants
from ..features import config

main_dictionary = config.FEATURES

# example for copy from my pc to brainik80a:
# rsync -avz -e ssh /home/yonatan/Pictures/collar/* yonatan@159.8.222.10:/home/yonatan/collar_classifier/collar_images/


def get_feature_data_ready(feature_name, crop=False):

    feature_dict = config.FEATURES[feature_name]
    source_dir = feature_dict['path_to_images']
    labels = feature_dict['labels']

    # edit_names.edit_dirs_names(source_dir)
    #
    # if crop:
    #     cropping.crop_figure_by_face_dir(source_dir)
    #
    # resize_and_save.resize_save_all_in_dir(feature_name, source_dir)

    preparing_txt_files.create_txt_files_from_different_directories(feature_name, source_dir, labels)
