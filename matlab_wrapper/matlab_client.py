__author__ = 'liorsabag'

import rpyc


class Engine(object):
    def __init__(self):
        self.conn = rpyc.connect("localhost", 18861)

    def __getattr__(self, name):
        try:
            # first check if this is an explicitly defined attribute on the server
            attr = getattr(self.conn.root, name)
        except:
            # otherwise pass it on to ml engine
            attr = self.conn.root.exposed_get_matlab_function
        return attr