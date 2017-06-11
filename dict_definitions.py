__author__ = 'jeremy'
#keep track of dicts and what they look like

#dict for keeping track of bbs (multiple bbs in an image - where and what are they )
#json file will have array of these dicts
tgdict = \
{   "dimensions_h_w_c": [360,640,3],  "filename": "/data/olympics/olympics/9908661.jpg", "annotations":
    [
        {
           "bbox_xywh": [89, 118, 64,44 ],
            "object": "car"
        } ,
        {
           "bbox_xywh": [11, 118, 64,34 ],
            "object": "singularity"
        }
    ]
}

