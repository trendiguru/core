FEATURES = {

    "sleeve_length": {
        "MODEL_FILE": "/home/yonatan/trendi/yonatan/resnet_50_dress_sleeve/ResNet-50-deploy.prototxt",
        "PRETRAINED": "/home/yonatan/dressSleeve_caffemodels/caffe_resnet50_snapshot_50_sgd_iter_10000.caffemodel",
        "labels": {
            'strapless': 0,
            'spaghetti_straps': 1,
            'regular_straps': 2,
            'sleeveless': 3,
            'cap_sleeve': 4,
            'short_sleeve': 5,
            'midi_sleeve': 6,
            'long_sleeve': 7
        },
        "relevant_items": []
    },

    "dress_length": {
        "MODEL_FILE": "/home/yonatan/trendi/yonatan/resnet_50_dress_length/ResNet-50-deploy.prototxt",
        "PRETRAINED": "/home/yonatan/dressLength_caffemodels/caffe_resnet50_snapshot_dress_length_3_categories_iter_10000.caffemodel",
        "labels": {
            'mini_length': 0,
            'midi_length': 1,
            'maxi_length': 2,
        },
        "relevant_items": []
    },

    "collar": {
        "MODEL_FILE": "/home/yonatan/trendi/yonatan/resnet_152_collar_type/ResNet-152-deploy.prototxt",
        "PRETRAINED": "/home/yonatan/collar_caffemodels/caffe_resnet152_snapshot_collar_9_categories_iter_2500.caffemodel",
        "labels": {
            'crew_neck': 0,
            'scoop_neck': 1,
            'v_neck': 2,
            'deep_v_neck': 3,
            'Henley_t_shirts': 4,
            'polo_collar': 5,
            'tie_neck': 6,
            'turtleneck': 7,
            'Hooded_T_Shirt': 8
        },
        "relevant_items": []
    },

    "style": {
        "MODEL_FILE": "/home/yonatan/trendi/yonatan/resnet_152_style/ResNet-152-deploy.prototxt",
        "PRETRAINED": "/home/yonatan/style_caffemodels/caffe_resnet152_snapshot_style_5_categories_iter_5000.caffemodel",
        "labels": {
            'swimsuit': 0,
            'sports': 1,
            'others': 2,
            'prom': 3,
            'bride_dress': 4
        },
        "relevant_items": []
    }
}
