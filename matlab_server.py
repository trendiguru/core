__author__ = 'liorsabag'

import rpyc
import matlab.engine

ENG = matlab.engine.start_matlab("-nodisplay")

class MyService(rpyc.Service):
    def on_connect(self):
        global ENG
        # code that runs when a connection is created
        # (to init the serivce, if needed)
        ENG = ENG or matlab.engine.start_matlab("-nodisplay")
        pass

    def on_disconnect(self):
        # code that runs when the connection has already closed
        # (to finalize the service, if needed)
        ENG.quit()
        pass

    def exposed_get_answer(self): # this is an exposed method
        return 42

    def get_question(self):  # while this method is not exposed
        return "what is the airspeed velocity of an unladen swallow?"


if __name__ == "__main__":
    from rpyc.utils.server import ThreadedServer
    t = ThreadedServer(MyService, port=18861)
    t.start()
    print "Started..."