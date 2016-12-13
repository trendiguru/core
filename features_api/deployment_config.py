DEPLOYMENTS =  {"collar":{"server":"http://37.58.101.173:8085"},
             "sleeve_length":{"server":"http://37.58.101.173:8081"},
             "length":{"server":"http://37.58.101.173:8083"},
             "style":{"server":"http://37.58.101.173:8089"},
             "dress_texture": {"server": "http://37.58.101.173:8082"},
             "gender":{"server":"http://37.58.101.173:8357"}}

#currently gender is started differently 
#gunicorn -b :8357 -w 5 -k gevent -n gender app:api
#and also it needs dlib, so you need to do pip install dlib in the container

# Generate url, by adding /{feature} to the end of server url
for feature, depl in DEPLOYMENTS.iteritems():
    depl["url"] = depl["server"].rstrip('/') + "/" + feature

