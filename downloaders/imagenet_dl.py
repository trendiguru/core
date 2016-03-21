# -*- coding: utf-8 -*-
__author__ = 'jeremy'

import sys
import os
from xml.etree import ElementTree
import requests

import sys
import os

synset_url = u"http://www.image-net.org/download/synset"
username = u""
accesskey = u""
filepath = u"images"

base_url = u"http://www.image-net.org/api/xml/"
structure_released = u"structure_released.xml"




def download_file(url, dst, params={}, debug=True):
    if debug:
        print u"downloading {0}...".format(dst),
    response = requests.get(url, params=params)
    content_type = response.headers["content-type"]
    if content_type.startswith("text"):
        raise TypeError("404 Error")
    with file(dst, "wb") as fp:
        fp.write(response.content)
    print "done."

def get_imagepath(wnid):
    return os.path.join(filepath, wnid + ".tar")

def main():
    if not os.path.exists(structure_released):
        print "The file {0} does not exist.".format(structure_released)
        download_file(base_url + structure_released,
                      structure_released)
    print "loading structure..."
    with file(structure_released, "r") as fp:
        tree = ElementTree.parse(fp)
        root = tree.getroot()
        release_data = root[0].text
        synsets = root[1]
        for child in synsets.iter():
            if len(child) > 0:
                continue
            wnid = child.attrib.get("wnid")
            imagepath = get_imagepath(wnid)
            if not os.path.exists(imagepath) or os.path.getsize(imagepath) == 0:
                params = {
                    "wnid": wnid,
                    "username": username,
                    "accesskey": accesskey,
                    "release": "latest",
                }
                download_file(synset_url, imagepath, params)

if __name__ == "__main__":
    sys.exit(main())
