import falcon
from falcon_cors import CORS
from . import falcon_jwt

from . import editor
from .temp_editor_users import USERS

login, auth_middleware = falcon_jwt.get_auth_objects(
    USERS.get, # get_users function
    "UPeQqp45xJeRgavxup8GzMTYTyDFwYND", # random secret
    3600, # expiration
    cookie_opts={"name": "my_auth_token",
                 "max_age": 86400,
                 "path": "/",
                 "http_only": True}
)


cors = CORS(allow_all_headers=True,
            allow_origins_list=['editor.trendi.guru', 'editor-dot-test-paper-doll.appspot.com'],
            allow_credentials_origins_list=['editor.trendi.guru', 'editor-dot-test-paper-doll.appspot.com'],
            allow_all_methods=True)

api = falcon.API(middleware=[cors.middleware, auth_middleware])

editor = editor.Editor()

api.add_route('/editor/images', editor)
api.add_route('/editor/images/{image_id}', editor)
api.add_route('/editor/images/{image_id}/people/{person_id}', editor)
api.add_route('/editor/images/{image_id}/people/{person_id}/items/{item_category}', editor)
api.add_route('/editor/images/{image_id}/people/{person_id}/items/{item_category}/similar_results/{results_collection}', editor)
api.add_route('/editor/images/{image_id}/people/{person_id}/items/{item_category}/similar_results/{results_collection}/{result_id}', editor)
api.add_route('/login', login)
