DEPLOYMENTS =  {"collar":{"server":"http://37.58.101.173:8085"},
             "sleeve_length":{"server":"http://37.58.101.173:8081"},
             "length":{"server":"http://37.58.101.173:8083"},
             "style":{"server":"http://37.58.101.173:8089"},
             "gender":{"server":"http://37.58.101.173:8357"}}

# Generate url, by adding /{feature} to the end of server url
for feature, depl in DEPLOYMENTS.iteritems():
    depl["url"] = depl["server"].rstrip('/') + "/" + feature

