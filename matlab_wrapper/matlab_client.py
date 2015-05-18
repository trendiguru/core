__author__ = 'liorsabag'

import rpyc


class Engine(object):
    def __init__(self, obtain_all=True):
        self.conn = rpyc.connect("localhost", 18861)
        self.obtain_all = obtain_all

    def rpyc_obtain_wrapper(self, remote_func):
        """
        RPyC usually returns proxy object - this is safer but causes problems, especially with numpy.
        Calling obtain fixes this. It makes a copy of the remote object (pickles and sends it over).
        Object has to be picklable...
        see: http://rpyc.readthedocs.org/en/latest/api/utils_classic.html
        :param remote_func:
        :return:
        """

        def wrapped_func(*args, **kwargs):
            return rpyc.utils.classic.obtain(remote_func(*args, **kwargs))

        return wrapped_func

    def __getattr__(self, name):
        try:
            # first check if this is an explicitly defined attribute on the server
            attr = getattr(self.conn.root, name)
        except:
            # otherwise pass it on to ml  engine
            attr = self.conn.root.exposed_get_matlab_function(name)
        return self.rpyc_obtain_wrapper(attr) if self.obtain_all else attr

    def get_pose_boxes_dict(self, path_to_image_or_url):
        d = self.conn.root.get_pose_boxes_dict(path_to_image_or_url)
        return rpyc.utils.classic.obtain(d) if d else None

