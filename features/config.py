FEATURES = {

    "sleeve_length": {
        "MODEL_FILE": "/usr/lib/python2.7/dist-packages/trendi/yonatan/resnet_50_dress_sleeve/ResNet-50-deploy.prototxt",
        "PRETRAINED": "/home/yonatan/dressSleeve_caffemodels/caffe_resnet50_snapshot_50_sgd_iter_10000.caffemodel",
        "path_to_images": "/home/yonatan/dresses_stuff/dress_sleeve_sets",
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

    "length": {
        "MODEL_FILE": "/usr/lib/python2.7/dist-packages/trendi/yonatan/resnet_50_dress_length/ResNet-50-deploy.prototxt",
        "PRETRAINED": "/home/yonatan/dressLength_caffemodels/caffe_resnet50_snapshot_dress_length_3_categories_iter_10000.caffemodel",
        "path_to_images": "/home/yonatan/dresses_stuff/dress_length_3_labels_sets",
        "labels": {
            'mini_length': 0,
            'midi_length': 1,
            'maxi_length': 2
        },
        "relevant_items": []
    },

    "collar": {
        "MODEL_FILE": "/usr/lib/python2.7/dist-packages/trendi/yonatan/resnet_152_collar_type/ResNet-152-deploy.prototxt",
        "PRETRAINED": "/home/yonatan/collar_caffemodels/caffe_resnet152_snapshot_collar_10_categories_iter_2500.caffemodel",
        "path_to_images": "/home/yonatan/collar_classifier/collar_images",
        "labels": {
            'crew_neck': 0,
            'scoop_neck': 1,
            'v_neck': 2,
            'deep_v_neck': 3,
            'Henley_t_shirts': 4,
            'polo_collar': 5,
            'tie_neck': 6,
            'turtleneck': 7,
            'Hooded_T_Shirt': 8,
            'strapless': 9
        },
        "relevant_items": []
    },

    "style": {
        "MODEL_FILE": "/usr/lib/python2.7/dist-packages/trendi/yonatan/resnet_152_style/ResNet-152-deploy.prototxt",
        "PRETRAINED": "/home/yonatan/style_caffemodels/caffe_resnet152_snapshot_style_5_categories_iter_5000.caffemodel",
        "path_to_images": "/home/yonatan/style_classifier/style_second_try/style_images",
        "labels": {
            'swimsuit': 0,
            'sports': 1,
            'others': 2,
            'prom': 3,
            'bride_dress': 4
        },
        "relevant_items": []
    },

    "dress_texture": {
        "MODEL_FILE": "/usr/lib/python2.7/dist-packages/trendi/yonatan/resnet_152_dress_texture/ResNet-152-deploy.prototxt",
        "PRETRAINED": "/home/yonatan/dressTexture_caffemodels/caffe_resnet152_snapshot_dress_texture_10_categories_iter_2500.caffemodel",
        "path_to_images": "/home/yonatan/dress_texture_classifier/dress_texture_images",
        "labels": {
            'one_color': 0,
            'applique': 1,
            'floral': 2,
            'square_pattern': 3,
            'dots': 4,
            'animal_print': 5,
            'zebra_print': 6,
            'stripes': 7,
            'chevron': 8,
            'colorblock': 9
        },
        "relevant_items": []
    }
}
