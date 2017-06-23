import falcon
from falcon_cors import CORS
from .sample import QuoteResource
# from multiprocessing import Pool
# process_pool = Pool(10)
from . import sync_images

cors = CORS(allow_all_headers=True, allow_all_origins=True, allow_all_methods=True)

api = falcon.API(middleware=[cors.middleware])

# images = sync_images.Images(process_pool)

images = sync_images.Images()

api.add_route('/images', images)
api.add_route('/quote', QuoteResource())
