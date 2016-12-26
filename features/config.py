FEATURES = {

    "sleeve_length": {
        "MODEL_FILE": "/data/production/caffemodels_and_protos/sleeve_length/ResNet-50-deploy.prototxt",
        "PRETRAINED": "/data/production/caffemodels_and_protos/sleeve_length/caffe_resnet50_snapshot_50_sgd_iter_10000.caffemodel",
        "path_to_images": "/data/production/caffemodels_and_protos/sleeve_length/images",
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
        "relevant_items": ['dress', 'top', 'shirt', 'blouse', 't-shirt']
    },

    "length": {
        "MODEL_FILE": "/data/production/caffemodels_and_protos/length/ResNet-50-deploy.prototxt",
        "PRETRAINED": "/data/production/caffemodels_and_protos/length/caffe_resnet50_snapshot_dress_length_3_categories_iter_10000.caffemodel",
        "path_to_images": "/data/production/caffemodels_and_protos/length/images",
        "labels": {
            'mini_length': 0,
            'midi_length': 1,
            'maxi_length': 2
        },
        "relevant_items": ['dress', 'skirt']
    },

    "collar": {
        "MODEL_FILE": "/data/production/caffemodels_and_protos/collar/ResNet-152-deploy.prototxt",
        "PRETRAINED": "/data/production/caffemodels_and_protos/collar/caffe_resnet152_snapshot_collar_10_categories_iter_2500.caffemodel",
        "path_to_images": "/data/production/caffemodels_and_protos/collar/images",
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
        "relevant_items": ['dress', 'top', 'shirt', 'blouse', 'sweater', 'sweatshirt', 't-shirt']
    },

    "style": {
        "MODEL_FILE": "/data/production/caffemodels_and_protos/style/ResNet-152-deploy.prototxt",
        "PRETRAINED": "/data/production/caffemodels_and_protos/style/caffe_resnet152_snapshot_style_5_categories_iter_5000.caffemodel",
        "path_to_images": "/data/production/caffemodels_and_protos/style/images",
        "labels": {
            'swimsuit': 0,
            'sports': 1,
            'others': 2,
            'prom': 3,
            'bride_dress': 4
        },
        "relevant_items": ['top', 'shirt', 'blouse', 'sweater', 'sweatshirt', 't-shirt']
    },

    "dress_texture": {
        "MODEL_FILE": "/data/production/caffemodels_and_protos/dress_texture/ResNet-152-deploy.prototxt",
        "PRETRAINED": "/data/production/caffemodels_and_protos/dress_texture/caffe_resnet152_snapshot_dress_texture_10_categories_iter_2500.caffemodel",
        "path_to_images": "/data/production/caffemodels_and_protos/dress_texture/images",
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
        "relevant_items": ['dress']
    }
}
