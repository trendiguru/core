# TODO: Make this generic etc..

# These are based on/necessary for shopstyle_dl script
db.categories.create_index("id", background=True)
db.shopstyle_US_Female_cache.create_index([("dl_version",1),("filter_params", 1)], background=True)
db.shopstyle_US_Female.create_index("p_hash", background=True)
db.shopstyle_US_Female.create_index("id", background=True)
# Used by NNSearch
db.shopstyle_US_Female.create_index("categories", background=True)
