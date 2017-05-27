# TODO: Make this generic etc..

db.categories.create_index("id", background=True)
db.shopstyle_US_Female_cache.create_index?
db.shopstyle_US_Female_cache.create_index([("dl_version",1),("filter_params", 1)], background=True)
db.shopstyle_US_Female.create_index("p_hash", background=True)
db.shopstyle_US_Female.create_index("id", background=True)
