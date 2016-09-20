import falcon

class StaticResource(object):
    def on_get(self, req, resp, path):
        print path
        # do some sanity check on the path
        resp.status = falcon.HTTP_200
        with open(path, 'r') as f:
            resp.body = f.read()
