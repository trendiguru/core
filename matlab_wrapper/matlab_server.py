__author__ = 'liorsabag'

import rpyc
import matlab.engine
import matlab
import numpy as np
import mat_2_py

from cv2 import imwrite


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

    def exposed_get_pose_boxes_dict(self, path_to_image):
        mat_boxes = ENG.get_pose_boxes_raw(path_to_image)
        np_boxes = np.array(mat_boxes, np.uint16)
        pose_dict = mat_2_py.translate_2_boxes(np_boxes)
        return pose_dict

    def exposed_call_matlab_function(self, func_name, *args):
        return getattr(ENG, func_name)(*args)


if __name__ == "__main__":
    from rpyc.utils.server import ThreadedServer
    t = ThreadedServer(MatlabServerService, port=18861)
    t.start()
    print "Ended..."