__author__ = 'liorsabag'

import rpyc
import matlab.engine
import numpy as np

ENG = matlab.engine.start_matlab("-nodisplay")

class MatlabServerService(rpyc.Service):
    def on_connect(self):
        global ENG
        # code that runs when a connection is created
        # (to init the serivce, if needed)
        ENG = ENG or matlab.engine.start_matlab("-nodisplay")
        print ENG
        pass

    def on_disconnect(self):
        # code that runs when the connection has already closed
        # (to finalize the service, if needed)
        pass

    def exposed_isprime(self, n):
        result = ENG.isprime(n)
        return result

    def exposed_call_matlab_function(self, func_name, *args):
        return getattr(ENG, func_name)(*args)


if __name__ == "__main__":
    from rpyc.utils.server import ThreadedServer
    t = ThreadedServer(MatlabServerService, port=18861)
    t.start()
    print "Ended..."