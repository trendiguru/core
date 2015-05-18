__author__ = 'liorsabag'

import urllib
import os

import rpyc
import numpy as np

import matlab.engine
import matlab
import mat_2_py
import Utils


os.chdir("/home/ubuntu/Dev/pose_estimation/20121128-pose-release-ver1.3/code-basic")


def init_ENG():
    ENG = matlab.engine.start_matlab("-nodisplay")
    ENG.chdir("/home/ubuntu/Dev/pose_estimation/20121128-pose-release-ver1.3/code-basic")
    ENG.init_pose(nargout=0)  # m file, adds mex paths (doesn't compile)
    return ENG

ENG = init_ENG()

class MatlabServerService(rpyc.Service):
    def on_connect(self):
        global ENG
        # code that runs when a connection is created
        # (to init the serivce, if needed)
        ENG = ENG or init_ENG()
        print ENG
        pass

    def on_disconnect(self):
        # code that runs when the connection has already closed
        # (to finalize the service, if needed)
        pass

    def exposed_get_pose_boxes_dict(self, path_to_image_or_url):
        if "://" in path_to_image_or_url:
            filename = Utils.format_filename(path_to_image_or_url)
            urllib.urlretrieve(path_to_image_or_url, "./images/" + filename)
        else:
            filename = path_to_image_or_url
        mat_boxes = ENG.get_pose_boxes_raw(path_to_image_or_url)
        np_boxes = np.array(mat_boxes, np.int16)
        pose_dict = mat_2_py.translate_2_boxes(np_boxes)
        if "://" in path_to_image_or_url:
            os.remove("./images/" + filename)
        return pose_dict

    def exposed_get_matlab_function(self, func_name):
        def wrapper(*args, **kwargs):
            retval = getattr(ENG, func_name)(*args, **kwargs)
            retval = np.array(retval) if type(retval) is matlab.mlarray else retval
        return wrapper


if __name__ == "__main__":
    from rpyc.utils.server import ThreadedServer
    t = ThreadedServer(MatlabServerService, port=18861,
                       protocol_config={"allow_public_attrs": True,
                                        "allow_all_attrs": True,
                                        "allow_pickle": True})
    t.start()
    print "Ended..."