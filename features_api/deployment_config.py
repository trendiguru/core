DEPLOYMENTS =  {"collar":{"server":"http://13.69.27.202:8085"},
             "sleeve_length":{"server":"http://13.69.27.202:8081"},
             "length":{"server":"http://13.69.27.202:8083"},
             "style":{"server":"http://13.69.27.202:8089"},
             "dress_texture": {"server": "http://13.69.27.202:8082"},
             "gender":{"server":"http://13.69.27.202:8357"}}

#ip used to be 37.58.101.173 on softlayer, now 13.69.27.202 on azure
# #currently gender is started differently
#gunicorn -b :8357 --workers 1 --timeout 300 trendi.features_api.gender_app:api
#and also it needs dlib, so you need to do pip install dlib in the container

#the rest are like 
#gunicorn -b :8085 --env GPU_DEVICE=1 --env FEATURES_JSON='["collar"]' -w 3 -k gevent -n collar --timeout 120 trendi.features_api.app:api

# Generate url, by adding /{feature} to the end of server url
for feature, depl in DEPLOYMENTS.iteritems():
    depl["url"] = depl["server"].rstrip('/') + "/" + feature

