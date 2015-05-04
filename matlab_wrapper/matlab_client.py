__author__ = 'liorsabag'

import rpyc


class Engine(object):
    def __init__(self):
        self.conn = rpyc.connect("localhost", 18861)

    def __getattr__(self, name):
        def func(*args):
            return self.conn.root.call_matlab_function(name, *args)
        return func