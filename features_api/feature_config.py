FEATURES =  {"collar":{"server":"http://37.58.101.173:8085"},
             "sleeve_length":{"server":"http://37.58.101.173:8081"},
             "dress_length":{"server":"http://37.58.101.173:8083"},
             "style":{"server":""},
             "gender":{"server":"http://37.58.101.173:8357"}}

# Generate url, by adding /{feature} to the end of server url
for feature, config in FEATURES.iteritems():
    config["url"] = config["server"].rstrip('/') + "/" + feature

