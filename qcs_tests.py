__author__ = 'Nadav Paz'

import unittest

import numpy as np

import Utils
from .constants import db


# UNIT TESTS - tests for every function

image_url = 'http://image.gala.de/v1/cms/JV/style-charlotte-casiraghi-4ge_9054537-ORIGINAL-imageGallery_standard.jpg?v=11800914'


class TestQcs(unittest.TestCase):
    def instead(self, func_name, *args, **kwargs):
        person_url = args[0]
        item_id = args[1]
        self.assertIsInstance(Utils.get_cv2_img_array(person_url), np.ndarray)
        self.assertIsInstance(db.images.find_one({'people.person.item_id': item_id}), dict)
