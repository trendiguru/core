import falcon
from falcon_cors import CORS

import editor

cors = CORS(allow_all_headers=True, allow_all_origins=True, allow_all_methods=True)

api = falcon.API(middleware=[cors.middleware])

editor = editor.Editor()

api.add_route('/editor/images', editor)
api.add_route('/editor/images/{image_id}', editor)
api.add_route('/editor/images/{image_id}/people/{person_id}', editor)
api.add_route('/editor/images/{image_id}/people/{person_id}/items/{item_category}', editor)
api.add_route('/editor/images/{image_id}/people/{person_id}/items/{item_category}/collections/{results_collection}', editor)
api.add_route('/editor/images/{image_id}/people/{person_id}/items/{item_category}/collections/{results_collection}/results/{result_id}', editor)

