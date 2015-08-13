__author__ = 'sergey'
import unittest
import json
from collections import Counter
import os
import shutil

import requests
import cv2

from category_tree import CatNode

"""
List of functions that not necessary to check:
- get_id
- print_c_node
- dict_is_suitable
- __repr__
"""
"""
---------------------------------
List of untested unctions:
- get_tree
- download_image
- push_url
- upload_new_tree
"""


class TddCatNode(unittest.TestCase):
    """
    tdd (Test-Driven Development)
    The main purpose of this class is to check easily a rightness of all functions of CatNode class.
    """

    def setUp(self):
        """
        The function allows us to put things in place before each test case.
        :return:None
        """
        self.main_tree = CatNode.get_tree()
        self.tree_dict = {"name": "root", "children": [
            {"name": "1", "children": [{"name": "1.1", "children": [], "attributes": []},
                                       {"name": "1.2", "children": [], "attributes": []},
                                       {"name": "1.3", "children": [], "attributes": []}], "attributes": []},
            {"name": "2", "children": [{"name": "2.1", "children": [], "attributes": []},
                                       {"name": "2.2", "children": [], "attributes": []},
                                       {"name": "2.3", "children": [], "attributes": []}], "attributes": []},
            {"name": "3", "children": [{"name": "3.1", "children": [], "attributes": []},
                                       {"name": "3.2", "children": [], "attributes": [{"name": "a1", "children": [
                                           {"name": "a1.1", "children": [],
                                            "attributes": []},
                                           {"name": "a1.2", "children": [],
                                            "attributes": []},
                                           {"name": "a1.3", "children": [],
                                            "attributes": []}]}]},
                                       {"name": "3.3", "children": [], "attributes": []}
                                       ], "attributes": []
             }], "attributes": []}
        self.tree_dict2 = {"name": "root", "children": [
            {"name": "1", "children": [{"name": "1.1", "children": [], "attributes": [{"name": "a1", "children": [
                {"name": "a1.1", "children": [],
                 "attributes": []},
                {"name": "a1.2", "children": [],
                 "attributes": []},
                {"name": "a1.3", "children": [],
                 "attributes": []}]}]},
                                       {"name": "1.2", "children": [], "attributes": [{"name": "b1", "children": [
                                           {"name": "b1.1", "children": [],
                                            "attributes": []},
                                           {"name": "b1.2", "children": [],
                                            "attributes": []},
                                           {"name": "b1.3", "children": [],
                                            "attributes": []}]}]},
                                       {"name": "1.3", "children": [], "attributes": []}], "attributes": []},
            {"name": "2", "children": [{"name": "2.1", "children": [], "attributes": []},
                                       {"name": "2.2", "children": [], "attributes": []},
                                       {"name": "2.3", "children": [], "attributes": []}], "attributes": []},
            {"name": "3", "children": [{"name": "3.1", "children": [], "attributes": []},
                                       {"name": "3.2", "children": [], "attributes": [{"name": "c1", "children": [
                                           {"name": "c1.1", "children": [],
                                            "attributes": []},
                                           {"name": "c1.2", "children": [],
                                            "attributes": []},
                                           {"name": "c1.3", "children": [],
                                            "attributes": []}]}]},
                                       {"name": "3.3", "children": [], "attributes": []}
                                       ], "attributes": []
             }], "attributes": [{"name": "a1", "children": [{"name": "d1.1", "children": [],
                                                             "attributes": []},
                                                            {"name": "d1.2", "children": [],
                                                             "attributes": []},
                                                            {"name": "d1.3", "children": [],
                                                             "attributes": []}]}]}
        self.tree_dict3 = {"name": "1", "children": [{"name": "1.1", "children": [], "attributes": []},
                                                     {"name": "1.2", "children": [], "attributes": []},
                                                     {"name": "1.3", "children": [], "attributes": []}],
                           "attributes": []}
        self.tree_dict4 = {"name": "single", "children": [], "attributes": []}
        self.tree_dict5 = \
            {"name": "1", "children": [{"name": "1.1", "children": [], "attributes": []},
                                       {"name": "1.2", "children": [], "attributes": []}], "attributes": [
                {"name": "a1", "attributes": [],
                 "children": [{"name": "a1.1",
                               "children": [],
                               "attributes": []},
                              {"name": "a1.2",
                               "children": [],
                               "attributes": []}]
                 }]}
        self.tree_dict6 = {
            u'children': [
                {u'children': [{u'children': [{u'attributes': [{u'children': [{u'description': u'front opening',
                                                                               u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10488.1.png',
                                                                               u'name': u'cardigan'},
                                                                              {
                                                                                  u'description': u'no front opening',
                                                                                  u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10442.1.png',
                                                                                  u'name': u'pullover'}],
                                                                u'name': u'style/opening'},
                                                               {u'children': [{u'description': u'covered neck',
                                                                               u'image': u'Neckline_GOLF.png',
                                                                               u'name': u'turtle'},
                                                                              {
                                                                                  u'description': u'round and close to the base of the neck',
                                                                                  u'image': u'Neckline_CLOSED.png',
                                                                                  u'name': u'crewneck'},
                                                                              {
                                                                                  u'description': u'v-shape, hits below neck and along chest',
                                                                                  u'image': u'Neckline_V.png',
                                                                                  u'name': u'v_neck'},
                                                                              {
                                                                                  u'description': u'3 edges that create square-shape at a point below the neck',
                                                                                  u'image': u'Neckline_SQUARE.png',
                                                                                  u'name': u'square'}],
                                                                u'description': u'shape of the sweater at the neck/chest',
                                                                u'name': u'neckline'}],
                                               u'description': u'knitted garment worn on upper body',
                                               u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10430.1.png',
                                               u'name': u'sweaters',
                                               u'tg_cat': u'mens-sweaters'},
                                              {u'attributes': [
                                                  {u'children': [{u'description': u'full length zipper closure',
                                                                  u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10445.1.png',
                                                                  u'name': u'zip sweatshirt'},
                                                                 {u'description': u'fully closed, no zipper',
                                                                  u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10446.1.png',
                                                                  u'name': u'pullover'}],
                                                   u'name': u'style'},
                                                  {u'children': [{
                                                      u'description': u'covering for the head, attached to the back',
                                                      u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10446.1.png',
                                                      u'name': u'hood'},
                                                      {u'description': u'does not have a hood',
                                                       u'image': u'https://webstores.activenetwork.com/school-software/raiders_cove_online/images/products/detail_343_raiders_cove_online_crew_neck_sweatshirt.jpg',
                                                       u'name': u'crewneck'}]}],
                                                  u'description': u'a loose, heavy shirt',
                                                  u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10433.2.png',
                                                  u'name': u'sweatshirts',
                                                  u'tg_cat': u'mens-sweatshirts'},
                                              {u'children': [{u'description': u'bare shoulders',
                                                              u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10475.1.png',
                                                              u'name': u'sleeveless',
                                                              u'tg_cat': u'mens-tees-and-tshirts'},
                                                             {u'description': u'sleeve along upper arm',
                                                              u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10478.1.png',
                                                              u'name': u'short sleeve',
                                                              u'tg_cat': u'mens-shortsleeve-shirts'},
                                                             {u'description': u'fully covered arm',
                                                              u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10474.1.p',
                                                              u'name': u'long sleeve',
                                                              u'tg_cat': u'mens-longsleeve-shirts'}],
                                               u'description': u'casual shirt, no collar or buttons',
                                               u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10478.1.png',
                                               u'name': u'tee shirts'},
                                              {u'attributes': [{u'children': [{
                                                  u'description': u'covered shoulders and covered upper portion of upper arm',
                                                  u'image': u'Sleeves_SHORT.png',
                                                  u'name': u'short sleeve'},
                                                  {
                                                      u'description': u'Shoulders + entire length of arm is covered',
                                                      u'image': u'Sleeves_LONG.png',
                                                      u'name': u'long sleeve'}],
                                                  u'name': u'sleeve length'},
                                                  {u'description': u'covered neck',
                                                   u'image': u'Neckline_COLLAR.png',
                                                   u'name': u'shirt collar'}],
                                                  u'description': u'collared shirt with buttons at neck',
                                                  u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10476.1.png',
                                                  u'name': u'polo shirts',
                                                  u'tg_cat': u'mens-polo-shirts'},
                                              {u'children': [
                                                  {u'description': u'dress shirt with sleeve to mid-upper arm',
                                                   u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10281.1.png',
                                                   u'name': u'short sleeve dress shirt'},
                                                  {
                                                      u'description': u'dress shirt with sleeve extending the full arm length',
                                                      u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10269.1.png',
                                                      u'name': u'long sleeve dress shirt'}],
                                                  u'description': u'button-down shirt with a collar',
                                                  u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10269.1.png',
                                                  u'name': u'dress shirt',
                                                  u'tg_cat': u'mens-dress-shirts'}],
                                u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10493.1.png',
                                u'name': u'tops'},
                               {u'children': [{u'children': [{u'description': u'narrow fitting',
                                                              u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10367.1.png',
                                                              u'name': u'skinny'},
                                                             {u'description': u'medium width leg',
                                                              u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10372.1.png',
                                                              u'name': u'straight leg'},
                                                             {u'description': u'loose-fitting, wide leg width',
                                                              u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10370.1.png',
                                                              u'name': u'baggy'}],
                                               u'description': u'denim pants',
                                               u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10367.1.png',
                                               u'name': u'jeans',
                                               u'tg_cat': u'mens-jeans'},
                                              {u'description': u'non-denim pants',
                                               u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10326.1.png',
                                               u'name': u'trousers',
                                               u'tg_cat': u'mens-pants'},
                                              {u'description': u'short, knee-length pants',
                                               u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10309.1.png',
                                               u'name': u'shorts',
                                               u'tg_cat': u'mens-shorts'}],
                                u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10326.1.png',
                                u'name': u'bottoms'},
                               {u'children': [{u'description': u'jacket with buttons and lapels, matching pants',
                                               u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10210.1.png',
                                               u'name': u'jacket'},
                                              {u'description': u'trousers matching the jacket',
                                               u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10205.1.png',
                                               u'name': u'pants'}],
                                u'description': u'matching jacket/pants',
                                u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10210.1.png',
                                u'name': u'suits',
                                u'tg_cat': u'mens-suits'},
                               {u'children': [{u'attributes': [{u'children': [{u'description': u'puffy look',
                                                                               u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10360.1.png',
                                                                               u'name': u'ski jacket',
                                                                               u'tg_cat': u'mens-jackets'},
                                                                              {u'description': u'made of wool',
                                                                               u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10361.1.png',
                                                                               u'name': u'wool coat',
                                                                               u'tg_cat': u'mens-wool-coats'},
                                                                              {u'description': u'made of leather',
                                                                               u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10349.1.png',
                                                                               u'name': u'leather jacket',
                                                                               u'tg_cat': u'mens-leather-and-suede-coats'},
                                                                              {u'description': u'made of fur',
                                                                               u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10343.1.png',
                                                                               u'name': u'fur coat',
                                                                               u'tg_cat': u'mens-jackets'},
                                                                              {u'description': u'made of denim',
                                                                               u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10345.1.png',
                                                                               u'name': u'jean jacket',
                                                                               u'tg_cat': u'mens-denim-jackets'}],
                                                                u'name': u'type/material'},
                                                               {u'children': [
                                                                   {u'description': u'headcovering at the back',
                                                                    u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10346.1.png',
                                                                    u'name': u'hood'},
                                                                   {
                                                                       u'description': u'no headcovering at the back',
                                                                       u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10356.1.png',
                                                                       u'name': u'no hood'}]},
                                                               {u'children': [{
                                                                   u'description': u'jacket hits from hip to mid-thigh',
                                                                   u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10340.1.png',
                                                                   u'name': u'short'},
                                                                   {
                                                                       u'description': u'jacket hits anywhere below mid-thigh to ankle',
                                                                       u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10342.1.png',
                                                                       u'name': u'long'}],
                                                                   u'name': u'length'}],
                                               u'description': u'outer garment with sleeves',
                                               u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10360.1.png',
                                               u'name': u'jackets',
                                               u'tg_cat': u'mens-jackets'},
                                              {
                                                  u'description': u'outer garment without sleeves, has open shoulders',
                                                  u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10578.2.png',
                                                  u'name': u'vests',
                                                  u'tg_cat': u'mens-vests'}],
                                u'description': u'clothes that are worn on the outside',
                                u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10360.1.png',
                                u'name': u'outerwear'}],
                 u'description': u"All men's clothing",
                 u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10210.1.png',
                 u'name': u"Men's clothing"},
                {
                    u'children': [{u'children': [{u'description': u'worn around the neck, with a suit or dress shirt',
                                                  u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10048.1.png',
                                                  u'name': u'tie'},
                                                 {u'description': u'worn around the neck, tied in a bow',
                                                  u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10008.1.png',
                                                  u'name': u'bowtie'}],
                                   u'name': u'suit accessories',
                                   u'tg_cat': u'mens-ties'},
                                  {u'description': u'worn around the waist of pants',
                                   u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10020.1.png',
                                   u'name': u'belts',
                                   u'tg_cat': u'mens-belts'},
                                  {u'children': [{u'description': u"small-med sized man's bag with handles",
                                                  u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10163.1.png',
                                                  u'name': u'briefcase',
                                                  u'tg_cat': u'mens-bags'},
                                                 {u'description': u'has shoulder straps',
                                                  u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10178.1.png',
                                                  u'name': u'backpack',
                                                  u'tg_cat': u'mens-backpacks'}],
                                   u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10178.1.png',
                                   u'name': u'bags'},
                                  {u'children': [{u'description': u'worn around the neck',
                                                  u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10075.1.png',
                                                  u'name': u'scarf',
                                                  u'tg_cat': u'mens-gloves-and-scarves'},
                                                 {u'description': u'worn on head',
                                                  u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10053.1.png',
                                                  u'name': u'hat',
                                                  u'tg_cat': u'mens-hats'},
                                                 {
                                                     u'description': u'hand coverings, with individually separated fingers',
                                                     u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10025.1.png',
                                                     u'name': u'gloves',
                                                     u'tg_cat': u'mens-gloves-and-scarves'}],
                                   u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10075.1.png',
                                   u'name': u'outerwear accessories'},
                                  {u'children': [{u'description': u'watch chain is metal',
                                                  u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10200.1.png',
                                                  u'name': u'metal'},
                                                 {u'description': u'leather',
                                                  u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10200.1.png',
                                                  u'name': u'leather'}],
                                   u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10200.1.png',
                                   u'name': u'watches',
                                   u'tg_cat': u'mens-watches-and-jewelry'},
                                  {u'children': [{u'description': u'closed-toe shoe with a tall shaft',
                                                  u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10650.1.png',
                                                  u'name': u'boots',
                                                  u'tg_cat': u'mens-boots'},
                                                 {u'description': u'closed toe shoes',
                                                  u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10644.1.png',
                                                  u'name': u'sneakers',
                                                  u'tg_cat': u'mens-sneakers'},
                                                 {u'description': u'usually leather',
                                                  u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10596.1.png',
                                                  u'name': u'dress shoes',
                                                  u'tg_cat': u'mens-lace-up-shoes'},
                                                 {u'description': u'closed-toe shoes',
                                                  u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10589.1.png',
                                                  u'name': u'loafers',
                                                  u'tg_cat': u'mens-slip-ons-shoes'},
                                                 {u'description': u'open-toed shoes, no backs',
                                                  u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10691.1.png',
                                                  u'name': u'flip flops',
                                                  u'tg_cat': u'mens-sandals'}],
                                   u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10644.1.png',
                                   u'name': u'shoes'}],
                    u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10048.1.png',
                    u'name': u"Men's accessories"}],
            u'description': u"Men's clothing and accessories",
            u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10210.1.png',
            u'name': u'Men'}
        self.tree_dict7 = {
            u'children': [
                {u'children': [{u'children': [{u'attributes': [{u'children': [{u'description': u'front opening',
                                                                               u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10488.1.png',
                                                                               u'name': u'cardigan'},
                                                                              {
                                                                                  u'description': u'no front opening',
                                                                                  u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10442.1.png',
                                                                                  u'name': u'pullover'}],
                                                                u'name': u'style/opening'},
                                                               {u'children': [{u'description': u'covered neck',
                                                                               u'image': u'Neckline_GOLF.png',
                                                                               u'name': u'turtle'},
                                                                              {
                                                                                  u'description': u'round and close to the base of the neck',
                                                                                  u'image': u'Neckline_CLOSED.png',
                                                                                  u'name': u'crewneck'},
                                                                              {
                                                                                  u'description': u'v-shape, hits below neck and along chest',
                                                                                  u'image': u'Neckline_V.png',
                                                                                  u'name': u'v_neck'},
                                                                              {
                                                                                  u'description': u'3 edges that create square-shape at a point below the neck',
                                                                                  u'image': u'Neckline_SQUARE.png',
                                                                                  u'name': u'square'}],
                                                                u'description': u'shape of the sweater at the neck/chest',
                                                                u'name': u'neckline'}],
                                               u'description': u'knitted garment worn on upper body',
                                               u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10430.1.png',
                                               u'name': u'sweaters',
                                               },
                                              {u'attributes': [
                                                  {u'children': [{u'description': u'full length zipper closure',
                                                                  u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10445.1.png',
                                                                  u'name': u'zip sweatshirt'},
                                                                 {u'description': u'fully closed, no zipper',
                                                                  u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10446.1.png',
                                                                  u'name': u'pullover'}],
                                                   u'name': u'style'},
                                                  {u'children': [{
                                                      u'description': u'covering for the head, attached to the back',
                                                      u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10446.1.png',
                                                      u'name': u'hood'},
                                                      {u'description': u'does not have a hood',
                                                       u'image': u'https://webstores.activenetwork.com/school-software/raiders_cove_online/images/products/detail_343_raiders_cove_online_crew_neck_sweatshirt.jpg',
                                                       u'name': u'crewneck'}]}],
                                                  u'description': u'a loose, heavy shirt',
                                                  u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10433.2.png',
                                                  u'name': u'sweatshirts',
                                                  u'tg_cat': u'mens-sweatshirts'},
                                              {u'children': [{u'description': u'bare shoulders',
                                                              u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10475.1.png',
                                                              u'name': u'sleeveless',
                                                              u'tg_cat': u'mens-tees-and-tshirts'},
                                                             {u'description': u'sleeve along upper arm',
                                                              u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10478.1.png',
                                                              u'name': u'short sleeve',
                                                              u'tg_cat': u'mens-shortsleeve-shirts'},
                                                             {u'description': u'fully covered arm',
                                                              u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10474.1.p',
                                                              u'name': u'long sleeve',
                                                              u'tg_cat': u'mens-longsleeve-shirts'}],
                                               u'description': u'casual shirt, no collar or buttons',
                                               u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10478.1.png',
                                               u'name': u'tee shirts'},
                                              {u'attributes': [{u'children': [{
                                                  u'description': u'covered shoulders and covered upper portion of upper arm',
                                                  u'image': u'Sleeves_SHORT.png',
                                                  u'name': u'short sleeve'},
                                                  {
                                                      u'description': u'Shoulders + entire length of arm is covered',
                                                      u'image': u'Sleeves_LONG.png',
                                                      u'name': u'long sleeve'}],
                                                  u'name': u'sleeve length'},
                                                  {u'description': u'covered neck',
                                                   u'image': u'Neckline_COLLAR.png',
                                                   u'name': u'shirt collar'}],
                                                  u'description': u'collared shirt with buttons at neck',
                                                  u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10476.1.png',
                                                  u'name': u'polo shirts',
                                                  u'tg_cat': u'mens-polo-shirts'},
                                              {u'children': [
                                                  {u'description': u'dress shirt with sleeve to mid-upper arm',
                                                   u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10281.1.png',
                                                   u'name': u'short sleeve dress shirt'},
                                                  {
                                                      u'description': u'dress shirt with sleeve extending the full arm length',
                                                      u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10269.1.png',
                                                      u'name': u'long sleeve dress shirt'}],
                                                  u'description': u'button-down shirt with a collar',
                                                  u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10269.1.png',
                                                  u'name': u'dress shirt',
                                                  u'tg_cat': u'mens-dress-shirts'}],
                                u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10493.1.png',
                                u'name': u'tops'},
                               {u'children': [{u'children': [{u'description': u'narrow fitting',
                                                              u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10367.1.png',
                                                              u'name': u'skinny'},
                                                             {u'description': u'medium width leg',
                                                              u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10372.1.png',
                                                              u'name': u'straight leg'},
                                                             {u'description': u'loose-fitting, wide leg width',
                                                              u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10370.1.png',
                                                              u'name': u'baggy'}],
                                               u'description': u'denim pants',
                                               u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10367.1.png',
                                               u'name': u'jeans',
                                               u'tg_cat': u'mens-jeans'},
                                              {u'description': u'non-denim pants',
                                               u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10326.1.png',
                                               u'name': u'trousers',
                                               u'tg_cat': u'mens-pants'},
                                              {u'description': u'short, knee-length pants',
                                               u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10309.1.png',
                                               u'name': u'shorts',
                                               u'tg_cat': u'mens-shorts'}],
                                u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10326.1.png',
                                u'name': u'bottoms'},
                               {u'children': [{u'description': u'jacket with buttons and lapels, matching pants',
                                               u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10210.1.png',
                                               u'name': u'jacket'},
                                              {u'description': u'trousers matching the jacket',
                                               u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10205.1.png',
                                               u'name': u'pants'}],
                                u'description': u'matching jacket/pants',
                                u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10210.1.png',
                                u'name': u'suits',
                                u'tg_cat': u'mens-suits'},
                               {u'children': [{u'attributes': [{u'children': [{u'description': u'puffy look',
                                                                               u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10360.1.png',
                                                                               u'name': u'ski jacket',
                                                                               u'tg_cat': u'mens-jackets'},
                                                                              {u'description': u'made of wool',
                                                                               u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10361.1.png',
                                                                               u'name': u'wool coat',
                                                                               u'tg_cat': u'mens-wool-coats'},
                                                                              {u'description': u'made of leather',
                                                                               u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10349.1.png',
                                                                               u'name': u'leather jacket',
                                                                               u'tg_cat': u'mens-leather-and-suede-coats'},
                                                                              {u'description': u'made of fur',
                                                                               u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10343.1.png',
                                                                               u'name': u'fur coat',
                                                                               u'tg_cat': u'mens-jackets'},
                                                                              {u'description': u'made of denim',
                                                                               u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10345.1.png',
                                                                               u'name': u'jean jacket',
                                                                               u'tg_cat': u'mens-denim-jackets'}],
                                                                u'name': u'type/material'},
                                                               {u'children': [
                                                                   {u'description': u'headcovering at the back',
                                                                    u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10346.1.png',
                                                                    u'name': u'hood'},
                                                                   {
                                                                       u'description': u'no headcovering at the back',
                                                                       u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10356.1.png',
                                                                       u'name': u'no hood'}]},
                                                               {u'children': [{
                                                                   u'description': u'jacket hits from hip to mid-thigh',
                                                                   u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10340.1.png',
                                                                   u'name': u'short'},
                                                                   {
                                                                       u'description': u'jacket hits anywhere below mid-thigh to ankle',
                                                                       u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10342.1.png',
                                                                       u'name': u'long'}],
                                                                   u'name': u'length'}],
                                               u'description': u'outer garment with sleeves',
                                               u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10360.1.png',
                                               u'name': u'jackets',
                                               u'tg_cat': u'mens-jackets'},
                                              {
                                                  u'description': u'outer garment without sleeves, has open shoulders',
                                                  u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10578.2.png',
                                                  u'name': u'vests',
                                                  u'tg_cat': u'mens-vests'}],
                                u'description': u'clothes that are worn on the outside',
                                u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10360.1.png',
                                u'name': u'outerwear'}],
                 u'description': u"All men's clothing",
                 u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10210.1.png',
                 u'name': u"Men's clothing"},
                {
                    u'children': [{u'children': [{u'description': u'worn around the neck, with a suit or dress shirt',
                                                  u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10048.1.png',
                                                  u'name': u'tie'},
                                                 {u'description': u'worn around the neck, tied in a bow',
                                                  u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10008.1.png',
                                                  u'name': u'bowtie'}],
                                   u'name': u'suit accessories',
                                   u'tg_cat': u'mens-ties'},
                                  {u'description': u'worn around the waist of pants',
                                   u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10020.1.png',
                                   u'name': u'belts',
                                   u'tg_cat': u'mens-belts'},
                                  {u'children': [{u'description': u"small-med sized man's bag with handles",
                                                  u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10163.1.png',
                                                  u'name': u'briefcase',
                                                  u'tg_cat': u'mens-bags'},
                                                 {u'description': u'has shoulder straps',
                                                  u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10178.1.png',
                                                  u'name': u'backpack',
                                                  u'tg_cat': u'mens-backpacks'}],
                                   u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10178.1.png',
                                   u'name': u'bags'},
                                  {u'children': [{u'description': u'worn around the neck',
                                                  u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10075.1.png',
                                                  u'name': u'scarf',
                                                  u'tg_cat': u'mens-gloves-and-scarves'},
                                                 {u'description': u'worn on head',
                                                  u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10053.1.png',
                                                  u'name': u'hat',
                                                  u'tg_cat': u'mens-hats'},
                                                 {
                                                     u'description': u'hand coverings, with individually separated fingers',
                                                     u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10025.1.png',
                                                     u'name': u'gloves',
                                                     u'tg_cat': u'mens-gloves-and-scarves'}],
                                   u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10075.1.png',
                                   u'name': u'outerwear accessories'},
                                  {u'children': [{u'description': u'watch chain is metal',
                                                  u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10200.1.png',
                                                  u'name': u'metal'},
                                                 {u'description': u'leather',
                                                  u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10200.1.png',
                                                  u'name': u'leather'}],
                                   u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10200.1.png',
                                   u'name': u'watches',
                                   u'tg_cat': u'mens-watches-and-jewelry'},
                                  {u'children': [{u'description': u'closed-toe shoe with a tall shaft',
                                                  u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10650.1.png',
                                                  u'name': u'boots',
                                                  u'tg_cat': u'mens-boots'},
                                                 {u'description': u'closed toe shoes',
                                                  u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10644.1.png',
                                                  u'name': u'sneakers',
                                                  u'tg_cat': u'mens-sneakers'},
                                                 {u'description': u'usually leather',
                                                  u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10596.1.png',
                                                  u'name': u'dress shoes',
                                                  u'tg_cat': u'mens-lace-up-shoes'},
                                                 {u'description': u'closed-toe shoes',
                                                  u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10589.1.png',
                                                  u'name': u'loafers',
                                                  u'tg_cat': u'mens-slip-ons-shoes'},
                                                 {u'description': u'open-toed shoes, no backs',
                                                  u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10691.1.png',
                                                  u'name': u'flip flops',
                                                  u'tg_cat': u'mens-sandals'}],
                                   u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10644.1.png',
                                   u'name': u'shoes'}],
                    u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10048.1.png',
                    u'name': u"Men's accessories"}],
            u'description': u"Men's clothing and accessories",
            u'image': u'https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10210.1.png',
            u'name': u'Men'}
        self.json_str = """
                    {
                      "name": "sleeves",
                      "description": "amount of coverage on the arm",
                      "image": "Sleeves_ASYMMETRIC.png",
                      "children": [
                        {
                          "name": "one_shoulder",
                          "description": "asymmetric with one shoulder covered, the other bare",
                          "image": "Sleeves_ASYMMETRIC.png"
                        },
                        {
                          "name": "halter_closed",
                          "description": "bare shoulders, straps wrap around neck",
                          "image": "Sleeves_halter_closed.png"
                        },
                        {
                          "name": "halter_open",
                          "description": "bare shoulders, straps are wide apart, wrap around neck",
                          "image": "Sleeves_halter_open.png"
                        },
                        {
                          "name": "strapless",
                          "description": "bare shoulders + collar; no sleeves or arm covering",
                          "image": "Sleeves_STRAPLESS.png"
                        },
                        {
                          "name": "strap",
                          "description": "thin straps over the shoulder/collar; sleeveless",
                          "image": "Sleeves_STRAP.png"
                        },
                        {
                          "name": "sleeveless",
                          "description": "partially bare shoulders, thick straps but no sleeves on arm",
                          "image": "Sleeves_SLEEVELESS.png"
                        },
                        {
                          "name": "short sleeve",
                          "description": "covered shoulders and covered upper portion of upper arm",
                          "image": "Sleeves_SHORT.png"
                        },
                        {
                          "name": "three-quarter sleeves",
                          "description": "covered shoulders and upper arm; bare lower arm",
                          "image": "Sleeves_THIRD_QUARTER.png"
                        },
                        {
                          "name": "long sleeve",
                          "description": "Shoulders and entire length of arm is covered",
                          "image": "Sleeves_LONG.png"
                        }
                      ]
                    }"""
        self.json_str2 = """{
    "categories": [
        {
            "name": "Men",
            "description": "Men's clothing and accessories",
            "image": "https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10210.1.png",
            "children": [
                {
                    "name": "Men's clothing",
                    "description": "All men's clothing",
                    "image": "https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10210.1.png",
                    "children": [
                        {
                            "name": "tops",
                            "image": "https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10493.1.png",
                            "children": [
                                {
                                    "name": "sweaters",
                                    "description": "knitted garment worn on upper body",
                                    "image": "https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10430.1.png",
                                    "attributes": [
                                        {
                                            "name": "style/opening",
                                            "children": [
                                                {
                                                    "name": "cardigan",
                                                    "description": "front opening",
                                                    "image": "https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10488.1.png"
                                                },
                                                {
                                                    "name": "pullover",
                                                    "description": "no front opening",
                                                    "image": "https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10442.1.png"
                                                }
                                            ]
                                        },
                                        {
                                            "name": "neckline",
                                            "description": "shape of the sweater at the neck/chest",
                                            "children": [
                                                {
                                                    "name": "turtle",
                                                    "description": "covered neck",
                                                    "image": "Neckline_GOLF.png"
                                                }
                                            ]
                                        }
                                    ]
                                },
                                {
                                    "name": "sweatshirts",
                                    "description": "a loose, heavy shirt",
                                    "image": "https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10433.2.png",
                                    "attributes": [
                                        {
                                            "name": "style",
                                            "children": [
                                                {
                                                    "name": "zip sweatshirt",
                                                    "description": "full length zipper closure",
                                                    "image": "https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10445.1.png"
                                                },
                                                {
                                                    "name": "pullover",
                                                    "description": "fully closed, no zipper",
                                                    "image": "https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10446.1.png"
                                                }
                                            ]
                                        },
                                        {
                                            "name": "???",
                                            "children": [
                                                {
                                                    "name": "hood",
                                                    "description": "covering for the head, attached to the back",
                                                    "image": "https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10446.1.png"
                                                },
                                                {
                                                    "name": "crewneck",
                                                    "description": "does not have a hood",
                                                    "image": "https://webstores.activenetwork.com/school-software/raiders_cove_online/images/products/detail_343_raiders_cove_online_crew_neck_sweatshirt.jpg"
                                                }
                                            ]
                                        }
                                    ]
                                },
                                {
                                    "name": "tee shirts",
                                    "description": "casual shirt, no collar or buttons",
                                    "image": "https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10478.1.png",
                                    "children": [
                                        {
                                            "name": "sleeveless",
                                            "description": "bare shoulders",
                                            "image": "https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10475.1.png"
                                        },
                                        {
                                            "name": "short sleeve",
                                            "description": "sleeve along upper arm",
                                            "image": "https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10478.1.png"
                                        },
                                        {
                                            "name": "long sleeve",
                                            "description": "fully covered arm",
                                            "image": "https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10474.1.p"
                                        }
                                    ]
                                },
                                {
                                    "name": "polo shirts",
                                    "description": "collared shirt with buttons at neck",
                                    "image": "https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10476.1.png",
                                    "attributes": [
                                        {
                                            "name": "sleeve length",
                                            "children": [
                                                {
                                                    "name": "short sleeve",
                                                    "description": "covered shoulders and covered upper portion of upper arm",
                                                    "image": "Sleeves_LONG.png"
                                                },
                                                {
                                                    "name": "long sleeve",
                                                    "description": "Shoulders + entire length of arm is covered",
                                                    "image": "Sleeves_LONG.png"
                                                }
                                            ]
                                        },
                                        {
                                            "name": "shirt collar",
                                            "description": "covered neck",
                                            "image": "Neckline_COLLAR.png"
                                        }
                                    ]
                                },
                                {
                                    "name": "dress shirt",
                                    "description": "button-down shirt with a collar",
                                    "image": "https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10269.1.png",
                                    "children": [
                                        {
                                            "name": "short sleeve dress shirt",
                                            "description": "dress shirt with sleeve to mid-upper arm",
                                            "image": "https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10281.1.png"
                                        },
                                        {
                                            "name": "long sleeve dress shirt",
                                            "description": "dress shirt with sleeve extending the full arm length",
                                            "image": "https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10269.1.png"
                                        }
                                    ]
                                }
                            ]
                        },
                        {
                            "name": "bottoms",
                            "image": "https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10326.1.png",
                            "children": [
                                {
                                    "name": "jeans",
                                    "description": "denim pants",
                                    "image": "https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10367.1.png",
                                    "children": [
                                        {
                                            "name": "skinny",
                                            "description": "narrow fitting",
                                            "image": "https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10367.1.png"
                                        },
                                        {
                                            "name": "straight leg",
                                            "description": "medium width leg",
                                            "image": "https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10372.1.png"
                                        },
                                        {
                                            "name": "baggy",
                                            "description": "loose-fitting, wide leg width",
                                            "image": "https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10370.1.png"
                                        }
                                    ]
                                },
                                {
                                    "name": "trousers",
                                    "description": "non-denim pants",
                                    "image": "https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10326.1.png"
                                },
                                {
                                    "name": "shorts",
                                    "description": "short, knee-length pants",
                                    "image": "https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10309.1.png"
                                }
                            ]
                        },
                        {
                            "name": "suits",
                            "description": "matching jacket/pants",
                            "image": "https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10210.1.png",
                            "children": [
                                {
                                    "name": "jacket",
                                    "description": "jacket with buttons and lapels, matching pants",
                                    "image": "https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10210.1.png"
                                },
                                {
                                    "name": "pants",
                                    "description": "trousers matching the jacket",
                                    "image": "https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10205.1.png"
                                }
                            ]
                        },
                        {
                            "name": "outerwear",
                            "description": "clothes that are worn on the outside",
                            "image": "https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10360.1.png",
                            "children": [
                                {
                                    "name": "jackets",
                                    "description": "outer garment with sleeves",
                                    "image": "https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10360.1.png",
                                    "attributes": [
                                        {
                                            "name": "type/material",
                                            "children": [
                                                {
                                                    "name": "ski jacket",
                                                    "description": "puffy look",
                                                    "image": "https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10360.1.png"
                                                },
                                                {
                                                    "name": "wool coat",
                                                    "description": "made of wool",
                                                    "image": "https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10361.1.png"
                                                }
                                            ]
                                        },
                                        {
                                            "name": "???",
                                            "children": [
                                                {
                                                    "name": "hood",
                                                    "description": "headcovering at the back",
                                                    "image": "https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10346.1.png"
                                                },
                                                {
                                                    "name": "no hood",
                                                    "description": "no headcovering at the back",
                                                    "image": "https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10356.1.png"
                                                }
                                            ]
                                        }
                                    ]
                                },
                                {
                                    "name": "vests",
                                    "description": "outer garment without sleeves, has open shoulders",
                                    "image": "https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10578.2.png"
                                }
                            ]
                        }
                    ]
                },
                {
                    "name": "Men's accessories",
                    "image": "https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10048.1.png",
                    "children": [
                        {
                            "name": "suit accessories",
                            "children": [
                                {
                                    "name": "tie",
                                    "description": "worn around the neck, with a suit or dress shirt",
                                    "image": "https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10048.1.png"
                                },
                                {
                                    "name": "bowtie",
                                    "description": "worn around the neck, tied in a bow",
                                    "image": "https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10008.1.png"
                                }
                            ]
                        },
                        {
                            "name": "belts",
                            "description": "worn around the waist of pants",
                            "image": "https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10020.1.png"
                        },
                        {
                            "name": "bags",
                            "image": "https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10178.1.png",
                            "children": [
                                {
                                    "name": "briefcase",
                                    "description": "small-med sized man's bag with handles",
                                    "image": "https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10163.1.png"
                                },
                                {
                                    "name": "backpack",
                                    "description": "has shoulder straps",
                                    "image": "https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10178.1.png"
                                }
                            ]
                        },
                        {
                            "name": "outerwear accessories",
                            "image": "https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10075.1.png",
                            "children": [
                                {
                                    "name": "scarf",
                                    "description": "worn around the neck",
                                    "image": "https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10075.1.png"
                                },
                                {
                                    "name": "hat",
                                    "description": "worn on head",
                                    "image": "https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10053.1.png"
                                },
                                {
                                    "name": "gloves",
                                    "description": "hand coverings, with individually separated fingers",
                                    "image": "https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10025.1.png"
                                }
                            ]
                        },
                        {
                            "name": "shoes",
                            "image": "https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10644.1.png",
                            "children": [
                                {
                                    "name": "boots",
                                    "description": "closed-toe shoe with a tall shaft",
                                    "image": "https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10650.1.png"
                                }
                            ]
                        }
                    ]
                }
            ]
        }
    ]
} """
        self.json_str3 = """{
    "name": "sleeves",
    "description": "amount of coverage on the arm",
    "image": "Sleeves_ASYMMETRIC.png",
    "children": [
        {
            "name": "one_shoulder",
            "description": "asymmetric with one shoulder covered, the other bare",
            "image": "Sleeves_ASYMMETRIC.png"
        },
        {
            "name": "halter_closed",
            "description": "bare shoulders, straps wrap around neck",
            "image": "Sleeves_halter_closed.png"
        },
        {
            "name": "halter_open",
            "description": "bare shoulders, straps are wide apart, wrap around neck",
            "image": "Sleeves_halter_open.png"
        },
        {
            "name": "strapless",
            "description": "bare shoulders + collar; no sleeves or arm covering",
            "image": "Sleeves_STRAPLESS.png"
        },
        {
            "name": "strap",
            "description": "thin straps over the shoulder/collar; sleeveless",
            "image": "Sleeves_STRAP.png"
        },
        {
            "name": "sleeveless",
            "description": "partially bare shoulders, thick straps but no sleeves on arm",
            "image": "Sleeves_SLEEVELESS.png",
            "attributes": [
                {
                    "name": "type/material",
                    "children": [
                        {
                            "name": "ski jacket",
                            "description": "puffy look",
                            "image": "https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10360.1.png"
                        },
                        {
                            "name": "wool coat",
                            "description": "made of wool",
                            "image": "https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10361.1.png"
                        }
                    ]
                }
            ]
        },
        {
            "name": "short sleeve",
            "description": "covered shoulders and covered upper portion of upper arm",
            "image": "Sleeves_SHORT.png"
        },
        {
            "name": "three-quarter sleeves",
            "description": "covered shoulders and upper arm; bare lower arm",
            "image": "Sleeves_THIRD_QUARTER.png"
        },
        {
            "name": "long sleeve",
            "description": "Shoulders and entire length of arm is covered",
            "image": "Sleeves_LONG.png"
        }
    ]
}"""
        self.json_str4 = """ {
    "categories": [
        {
            "name": "Men",
            "description": "Men's clothing and accessories",
            "image": "https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10210.1.png",
            "children": [
                {
                    "name": "Men's clothing",
                    "description": "All men's clothing",
                    "image": "https://s3.eu-central-1.amazonaws.com/fashion-category-images/name",
                    "children": [
                        {
                            "name": "tops",
                            "image": "https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10493.1.png",
                            "tg_cat": "mens-sweaters",
                            "children": [
                                {
                                    "name": "sweaters",
                                    "description": "knitted garment worn on upper body",
                                    "image": "https://s3.eu-central-1.amazonaws.com/fashion-category-images/name",
                                    "children": [
                                        {
                                            "description": "knitted garment worn on upper body",
                                            "image": "https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10430.1.png",
                                            "name": "style/opening",
                                            "children": [
                                                {
                                                    "name": "cardigan",
                                                    "description": "front opening",
                                                    "image": "https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10488.1.png"
                                                },
                                                {
                                                    "name": "pullover",
                                                    "description": "no front opening",
                                                    "image": "https://s3.eu-central-1.amazonaws.com/fashion-category-images/name"
                                                }
                                            ]
                                        },
                                        {
                                            "name": "neckline",
                                            "image": "https://s3.eu-central-1.amazonaws.com/fashion-category-images/name",
                                            "description": "shape of the sweater at the neck/chest",
                                            "children": [
                                                {
                                                    "name": "turtle",
                                                    "description": "covered neck",
                                                    "image": "https://s3.eu-central-1.amazonaws.com/fashion-category-images/name"
                                                }
                                            ]
                                        }
                                    ]
                                }
                            ]
                        }
                    ]
                }
            ]
        }
    ]
} """

    # ---Block_of_tests_to_standard_CatNode_functions--- #

    def test_init_(self):
        """
        The function checks if __init__ function really builds
        :return:
        """
        tree_dict = self.tree_dict
        # check if tree is build:
        new_tree = CatNode(**tree_dict)
        self.assertEqual([child.name for child in new_tree.children], ["1", "2", "3"])
        for sub_tree in new_tree.children:
            self.assertEqual([child.name for child in sub_tree.children], [sub_tree.name + ".1", sub_tree.name + ".2",
                                                                           sub_tree.name + ".3"])
        # check if tree's attributes were build:
        attr_child = new_tree.children[2].children[1]
        self.assertEqual(attr_child.attributes[0].name, "a1")
        self.assertEqual([child.name for child in attr_child.attributes[0].children], ["a1.1", "a1.2", "a1.3"])

    def test_add_child(self):
        new_tree = CatNode(**self.tree_dict)
        new_tree.add_child(CatNode(**{"name": "new_child"}))
        self.assertEqual(new_tree.children[3].name, "new_child")

    def test_add_child(self):
        new_tree = CatNode(**self.tree_dict)
        new_tree.add_child(CatNode(**{"name": "new_child"}))
        self.assertEqual(new_tree.children[3].name, "new_child")

    def test_add_attribute(self):
        new_tree = CatNode(**self.tree_dict)
        new_tree.add_attributes(CatNode(**{"name": "new_attribute"}))
        self.assertEqual(new_tree.attributes[0].name, "new_attribute")

    def test_count_amount_of_primary_nodes(self):
        new_tree = CatNode(**self.tree_dict)
        new_tree.complicate()
        self.assertEqual(new_tree.count_amount_of_primary_nodes(), 13)

    def test_count_amount_of_attr_nodes(self):
        new_tree = CatNode(**self.tree_dict)
        new_tree.complicate()
        self.assertEqual(new_tree.count_amount_of_attr_nodes(), 3)
        new_tree = CatNode(**self.tree_dict2)
        new_tree.complicate()
        self.assertEqual(new_tree.count_amount_of_attr_nodes(), 30)

    def test_size(self):
        new_tree = CatNode(**self.tree_dict)
        self.assertEqual(new_tree.size(), 13)
        new_tree = CatNode(**self.tree_dict2)
        new_tree.complicate()
        self.assertEqual(new_tree.size(), 67)

    def test_count_attributes(self):
        new_tree = CatNode(**self.tree_dict)
        self.assertEqual(new_tree.count_attributes(), 1)
        new_tree = CatNode(**self.tree_dict2)
        self.assertEqual(new_tree.count_attributes(), 4)

    def test_connect_to_leafs(self):
        new_tree = CatNode(**self.tree_dict)
        sub_tree = CatNode(**self.tree_dict3)
        sub_tree2 = CatNode(**self.tree_dict4)
        new_tree.connect_to_leafs([sub_tree, sub_tree2])
        self.assertEqual(new_tree.size(), 58)

    def test_connect_sub_trees(self):
        new_tree = CatNode(**self.tree_dict)
        sub_tree = CatNode(**self.tree_dict3)
        sub_tree2 = CatNode(**self.tree_dict4)
        sub_tree3 = CatNode(**self.tree_dict)
        new_tree.connect_sub_trees([sub_tree, sub_tree2])
        self.assertEqual(new_tree.size(), 40)
        new_tree = CatNode(**self.tree_dict)
        new_tree.connect_sub_trees([sub_tree, sub_tree2, sub_tree3])
        self.assertEqual(new_tree.size(), 364)

    def test_complicate(self):
        new_tree = CatNode(**self.tree_dict5)
        new_tree.complicate()
        self.assertEqual(new_tree.children[0].children[0].name, new_tree.children[1].children[0].name)
        self.assertEqual(new_tree.children[0].children[1].name, new_tree.children[1].children[1].name)
        new_tree = CatNode(**self.tree_dict)
        new_tree.complicate()
        self.assertEqual(new_tree.size(), 16)
        self.assertEqual(new_tree.count_amount_of_primary_nodes() + new_tree.count_amount_of_attr_nodes(), 16)
        new_tree = CatNode(**self.tree_dict2)
        new_tree.complicate()
        self.assertEqual(new_tree.size(), 67)
        self.assertEqual(new_tree.count_amount_of_primary_nodes() + new_tree.count_amount_of_attr_nodes(), 67)

    def test_copy(self):
        new_tree = CatNode(**self.tree_dict5)
        new_tree_copy = CatNode()
        new_tree.complicate()
        new_tree_copy.complicate()
        new_tree_copy.copy(new_tree)
        self.assertEqual(new_tree.name, new_tree_copy.name)
        for i in range(0, 1):
            self.assertEqual(new_tree.children[i].name, new_tree_copy.children[i].name)
            for j in range(0, 1):
                self.assertEqual(new_tree.children[i].children[j].name, new_tree_copy.children[i].children[j].name)

    def test_from_str(self):
        # Build tree from json structure:
        new_tree = CatNode.from_str(self.json_str)[0]
        self.assertEqual(new_tree.size(), 10)
        self.assertEqual(new_tree.name, "sleeves")
        self.assertEqual(new_tree.children[8].name, "long sleeve")
        self.assertEqual(new_tree.children[0].name, "one_shoulder")
        # Build tree from json structure (with "categories" root)
        new_tree = CatNode.from_str(self.json_str2)[0]
        self.assertEqual(new_tree.size(), 40)
        new_tree.complicate()
        self.assertEqual(new_tree.size(), new_tree.count_amount_of_attr_nodes() +
                         new_tree.count_amount_of_primary_nodes())
        self.assertEqual(new_tree.children[0].children[1].children[2].name, "shorts")

    def test_to_struct(self):
        new_tree = CatNode(**self.tree_dict)
        tree_dict = {}
        new_tree.to_struct(tree_dict)
        # check if after a function implementation the obtainable tree is not changed:
        self.assertEqual(new_tree.size(), 13)
        # check if tree_dict is right:
        counter1 = 1
        counter2 = 1
        for child in tree_dict["children"]:
            self.assertIsNotNone(child.get("name"))
            self.assertIsNotNone(child.get("children"))
            self.assertEqual(child["name"], str(counter1))
            for grand_child in child["children"]:
                self.assertIsNotNone(grand_child.get("name"))
                self.assertIsNone(grand_child.get("children"))
                self.assertEqual(grand_child["name"], str(counter1) + "." + str(counter2))
                counter2 += 1
            counter2 = 1
            counter1 += 1
        tree_dict = {}
        new_tree.complicate()
        new_tree.to_struct(tree_dict)
        # check if after a function implementation the obtainable tree is not changed:
        self.assertEqual(new_tree.size(), 16)
        self.assertIsNotNone(tree_dict.get("name"))
        self.assertIsNotNone(tree_dict.get("children"))
        self.assertEqual(tree_dict["name"], "root")

    def test_to_json(self):
        new_tree = CatNode(**self.tree_dict)
        # we check if we have got a correct json after an implementation of to_js function:
        self.assertRaises(json.loads(new_tree.to_js()))

    def test_correct_levels(self):
        new_tree = CatNode(**self.tree_dict)
        some_node1 = new_tree.children[0].children[0]
        some_node2 = new_tree.children[0].children[1]
        some_node3 = new_tree.children[0].children[2]
        some_node4 = new_tree.children[1].children[0]
        some_node5 = new_tree.children[1]
        some_node1.level = -2
        some_node2.level = 980
        some_node3.level = 903
        some_node4.level = 0
        some_node4.level = "as"
        new_tree.correct_levels()
        self.assertEqual(some_node1.level, 3)
        self.assertEqual(some_node2.level, 3)
        self.assertEqual(some_node3.level, 3)
        self.assertEqual(some_node4.level, 3)
        self.assertEqual(some_node5.level, 2)

    def test_full_tree_to_js(self):
        # find number of all nodes in a previous json tree:
        new_tree = CatNode.from_str(self.json_str2)
        json_list1 = self.json_str2.split()
        json_size1 = Counter(json_list1)['"name":']  # The problem is if some of the node descriptions contains "name":.
        json_tree = CatNode.full_tree_to_js(new_tree)
        json_list2 = json_tree.split()
        json_size2 = Counter(json_list2)['"name":']
        f = open("temp.txt", "w")
        f.write(json_tree)
        f.close()
        self.assertEqual(json_size1, json_size2)

    # ---Block_of_tests_to_"change_urls_in_tree"_functions--- #

    def test_apply_to_all_nodes(self):
        def temp_func(c_node):
            c_node.name = "little Billy"

        new_tree = CatNode(**self.tree_dict)
        new_tree.apply_to_all_nodes(temp_func)
        for child in new_tree.children:
            self.assertEqual(child.name, "little Billy")
            for grand_child in child.children:
                self.assertEqual(grand_child.name, "little Billy")

    def test_is_only_name(self):
        str1 = "Sleeves_halter_open.png"
        str2 = "https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10210.1.png"
        str3 = "https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10493.1.png"
        str4 = "Sleeves_short.png"
        str5 = "Sleeves_LONG.png"
        self.assertFalse(CatNode.is_only_name(str2))
        self.assertFalse(CatNode.is_only_name(str3))
        self.assertTrue(CatNode.is_only_name(str1))
        self.assertTrue(CatNode.is_only_name(str4))
        self.assertTrue(CatNode.is_only_name(str5))

    def test_get_name(self):
        str1 = "Sleeves_halter_open.png"
        str2 = "https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10210.1.png"
        str3 = "https://d336s8nt4p00c3.cloudfront.net/tags/de/res/60/10493.1.png"
        str4 = "Sleeves_short.png"
        str5 = "Sleeves_LONG.png"
        self.assertEqual(CatNode.get_name(str1), "Sleeves_halter_open.png")
        self.assertEqual(CatNode.get_name(str2), "10210.1.png")
        self.assertEqual(CatNode.get_name(str3), "10493.1.png")
        self.assertEqual(CatNode.get_name(str4), "Sleeves_short.png")
        self.assertEqual(CatNode.get_name(str5), "Sleeves_LONG.png")

    def test_update_url(self):
        new_tree = CatNode.from_str(self.json_str)
        new_tree[0].apply_to_all_nodes(CatNode.update_url,
                                       "https://s3.eu-central-1.amazonaws.com/fashion-category-images/",
                                       ["www", "wwww", "wwwww"])
        struct_dict = {}
        new_tree[0].to_struct(struct_dict)
        self.assertEqual(struct_dict["image"][:-len(CatNode.get_name(struct_dict["image"]))],
                         "https://s3.eu-central-1.amazonaws.com/fashion-category-images/")
        for child in struct_dict["children"]:
            self.assertEqual(child["image"][:-len(CatNode.get_name(child["image"]))],
                             "https://s3.eu-central-1.amazonaws.com/fashion-category-images/")

    # ------------------ block_of_find_right_answer_functions ---------------------------
    def test_head(self):
        CatNode.cat_tree = CatNode(**self.tree_dict6)
        self.assertTrue(CatNode.cat_tree.check_tree())
        CatNode.cat_tree = CatNode(**self.tree_dict7)
        self.assertFalse(CatNode.cat_tree.check_tree())

    def test_upload_new_tree(self):

        def name_from_url(url):
            return os.path.basename(url.split('?')[0])

        def _download_image(url, destination_dir):
            name = name_from_url(url)
            dest_path = name_from_url(url)
            try:
                result = requests.get(url)
                if result.status_code is not 200:
                    raise IOError("Code is not 200")
                else:
                    with open(dest_path, "wb") as f:
                        for chunk in result.iter_content():
                            f.write(chunk)
                    f.close()
                    img = cv2.imread(name, cv2.CV_LOAD_IMAGE_COLOR)
                    cv2.imwrite(destination_dir + "//" + name, img)
                    os.remove(name)
            except:
                return False
            return True

        if not os.path.exists("temp_folder"):
            os.makedirs("temp_folder")
        js_tree = self.json_str4
        CatNode.__upload_new_tree__(js_tree,
                                    "https://s3.eu-central-1.amazonaws.com/fashion-category-images/", _download_image,
                                    "C:\Users\sergey\PycharmProjects\\bitbucket_projects\core\\temp_folder")
        self.assertEqual(len(os.listdir("temp_folder")), 4)
        shutil.rmtree("temp_folder")

    def test_find_ans(self):
        CatNode.reset_id()
        tree = CatNode(**self.tree_dict6)
        print type(tree.find_ans([34, 35]).id)
        self.assertEqual(tree.find_ans([34, 35]).id, 33)
        self.assertEqual(tree.find_ans([]).id, 1)
        self.assertEqual(tree.find_ans([29, 29]).id, 29)
        self.assertEqual(tree.find_ans([72, 70]).id, 57)
        self.assertEqual(tree.find_ans([75, 74, 70, 71]).id, 57)
        self.assertEqual(tree.find_ans([69, 69, 69]).id, 69)
        self.assertEqual(tree.find_ans([36, 35, 43, 42, 72, 74, 69]).id, 2)
        self.assertEqual(tree.find_ans([35]).id, 1)
        self.assertEqual(tree.find_ans([72, 72, 72, 72, 43, 43, 43, 43]).id, 1)
        self.assertEqual(tree.find_ans([75, 75, 75]).id, 75)
        self.assertEqual(tree.find_ans([72, 75, 74, 70, 71]).id, 72)

    def test_build_bucket(self):
        CatNode.reset_id()
        tree = CatNode(**self.tree_dict6)
        CatNode.cat_tree = tree
        self.assertEqual(CatNode.build_bucket(69).key, 69)
        self.assertEqual(CatNode.build_bucket(71).key, 69)
        self.assertEqual(CatNode.build_bucket(51), None)
        self.assertEqual(CatNode.build_bucket(59).key, 58)
        self.assertEqual(CatNode.build_bucket(40).key, 39)
        self.assertEqual(CatNode.build_bucket(1), None)

    def test_check_ans(self):
        CatNode.reset_id()
        tree = CatNode(**self.tree_dict6)
        CatNode.cat_tree = tree
        res_mat = CatNode.check_ans([[70, 71, 19999, "as", 57], [63, 62, -8], [59, 58, 60, 40, 39]])
        self.assertEqual(res_mat[0][0], 71)
        self.assertEqual(res_mat[1][0], 63)
        self.assertEqual(res_mat[2][0], 60)
        self.assertEqual(res_mat[2][1], 39)

    def test_push_to_bucket(self):
        CatNode.reset_id()
        tree = CatNode(**self.tree_dict6)
        CatNode.cat_tree = tree
        b = CatNode.build_bucket(33)
        CatNode.push_to_bucket(b, [34, 35, 36])
        self.assertEqual(b.content, [33, 34, 35, 36])
        CatNode.push_to_bucket(b, [-234, 74])
        self.assertEqual(b.content, [33, 34, 35, 36])

    def test_determine_final_categories(self):
        CatNode.reset_id()
        tree = CatNode(**self.tree_dict6)
        CatNode.cat_tree = tree
        self.assertEqual(CatNode.determine_final_categories([[1, 73, 70, 59, 60, -5, 39],
                                                             [1, 74, 71, 59, 60, -3, 39],
                                                             [1, 75, 71, 60, 60, "be", 39]]), ['60', '71', '39'])

    def test_root(self):
        tree = CatNode(**self.tree_dict6)
        self.assertEqual(tree.root().name, "Men")

    def test___is_common__(self):
        CatNode.reset_id()
        tree = CatNode(**self.tree_dict6)
        CatNode.cat_tree = tree
        self.assertTrue(CatNode.__is_common__(tree.find_by_id(57), [tree.find_by_id(id) for id in [75, 74, 72,
                                                                                                   69, 70, 71]]))
        self.assertTrue(CatNode.__is_common__(tree.find_by_id(2), [tree.find_by_id(id) for id in [41, 42]]))
        self.assertFalse(CatNode.__is_common__(tree.find_by_id(57), [tree.find_by_id(id) for id in [75, 74, 72,
                                                                                                    69, 70, 71, 1, 2]]))
        self.assertFalse(CatNode.__is_common__(tree.find_by_id(33), [tree.find_by_id(id) for id in [32]]))
        self.assertTrue(CatNode.__is_common__(tree.find_by_id(1), [tree.find_by_id(id) for id in [75, 36, 42]]))

    def test_common_root(self):
        CatNode.reset_id()
        tree = CatNode(**self.tree_dict6)
        CatNode.cat_tree = tree
        self.assertEqual(tree.common_root([tree.find_by_id(id) for id in [75, 42]]).id, 1)
        self.assertEqual(tree.common_root([tree.find_by_id(id) for id in [36, 35]]).id, 33)
        self.assertEqual(tree.common_root([tree.find_by_id(id) for id in [75, 74]]).id, 72)
        self.assertEqual(tree.common_root([tree.find_by_id(id) for id in [36, 37, 41]]).id, 2)
        self.assertEqual(tree.common_root([tree.find_by_id(id) for id in [41, 42, 74]]).id, 1)

    def test___fbi__(self):
        CatNode.reset_id()
        tree = CatNode(**self.tree_dict6)
        CatNode.cat_tree = tree
        # find a single node by its id:
        self.assertTrue(tree.find_by_id(32).id == 32 and type(tree.find_by_id(32)) is CatNode)
        self.assertTrue(tree.find_by_id(69).id == 69 and type(tree.find_by_id(69)) is CatNode)
        self.assertTrue(tree.find_by_id(1).id == 1 and type(tree.find_by_id(1)) is CatNode)
        self.assertTrue(tree.find_by_id(2).id == 2 and type(tree.find_by_id(2)) is CatNode)
        self.assertTrue(tree.find_by_id(74).id == 74 and type(tree.find_by_id(74)) is CatNode)
        print type(tree.find_by_id(74, True))
        self.assertTrue(tree.find_by_id(74, True) == '74' and type(tree.find_by_id(74, True)) is str)
        self.assertTrue(tree.find_by_id(-32) is None)
        # find a list of nodes by their id:
        self.assertTrue(tree.find_by_id([75, 42, 37], True) == ['37', '42', '75'])
        self.assertTrue([node.id for node in tree.find_by_id([75, 42, 37])] == [37, 42, 75])
        self.assertTrue([node.id for node in tree.find_by_id([75, 74, 70, 71])] == [70, 71, 74, 75])
        self.assertTrue([node.id for node in tree.find_by_id([2, 42, 41, 57])] == [2, 41, 42, 57])
        self.assertTrue([node.id for node in tree.find_by_id([69, 41, -1])] == [41, 69])

    def test_build_path(self):
        CatNode.reset_id()
        tree = CatNode(**self.tree_dict6)
        CatNode.cat_tree = tree
        self.assertEqual([node.id for node in CatNode.build_path(tree.find_by_id(75))], [1, 57, 72, 75])
        self.assertEqual([node.id for node in CatNode.build_path(tree.find_by_id(37))], [1, 2, 32, 37])

# ---Block_of_upload_new_tree------------------------------------------------#

if __name__ == '__main__':
    CatNode.get_tree()
    unittest.main()