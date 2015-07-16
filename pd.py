__author__ = 'jeremy'
# paperdoll run from matlab
# this needs to run on server that
# has matlab running w. opencv3
#which is currently 'mightili.trendi.guru'


import matlab.engine


def get_parse_mask(image_filename):
    eng = matlab.engine.start_matlab()
    result = eng.pd(image_filename)


if __name__ == "__main__":
    get_parse_mask('img.jpg')
