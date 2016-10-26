__author__ = 'jeremy'

import time
#from trendi import multilabel_from_binaries
#from trendi import multilabel_from_binaries2
from trendi.paperdoll import binary_multilabel_falcon_client as bmfc
from trendi.paperdoll import binary_multilabel_falcon_client2 as bmfc2
from trendi.paperdoll import binary_multilabel_falcon_client3 as bmfc3
from trendi import Utils

import logging
logging.basicConfig(level=logging.DEBUG)

def get_mlb_output(url_or_np_array):

    dic1 = bmfc.mlb(url_or_np_array)
    if not dic1['success']:
        logging.debug('nfc mlb not a success')
        return False
    output1 = dic1['output']
    dic2 = bmfc2.mlb(url_or_np_array)
    if not dic2['success']:
        logging.debug('nfc mlb2 not a success')
        return False
    output2 = dic2['output']
    dic3 = bmfc3.mlb(url_or_np_array)
    if not dic3['success']:
        logging.debug('nfc mlb3 not a success')
        return False
    output3 = dic3['output']
    output = output1+output2+output3
    return output


if __name__ == "__main__":
    urls = ['https://s-media-cache-ak0.pinimg.com/236x/ce/64/a0/ce64a0dca7ad6d609c635432e9ae1413.jpg',  #bags
            'http://pinmakeuptips.com/wp-content/uploads/2015/02/1.4.jpg',
            'https://s-media-cache-ak0.pinimg.com/564x/9a/9d/f7/9a9df7455232035c6284ad1961816fd8.jpg',
            'http://2.bp.blogspot.com/-VmiQlqdliyE/U9nyto2L1II/AAAAAAAADZ8/g30i4j_YZeI/s1600/310714+awayfromblue+kameleon+shades+chambray+shift+dress+converse+louis+vuitton+neverfull+mbmj+scarf.png',
            'https://s-media-cache-ak0.pinimg.com/236x/1b/31/fd/1b31fd2182f0243ebc97ca115f04f131.jpg',
            'http://healthsupporters.com/wp-content/uploads/2013/10/belt_2689094b.jpg' ,
            'http://static1.businessinsider.com/image/53c96c90ecad04602086591e-480/man-fashion-jacket-fall-layers-belt.jpg', #belts
            'http://gunbelts.com/media/wysiwyg/best-gun-belt-width.jpg',
            'https://i.ytimg.com/vi/5-jWNWUQdFQ/maxresdefault.jpg'
            ]

    start_time=time.time()
    for url in urls:
        image = Utils.get_cv2_img_array(url)
        output = get_mlb_output(image)
#        output = get_mlb_output(url)
#        output1 = multilabel_from_binaries.get_multiple_single_label_outputs(url)
#        output2 = multilabel_from_binaries2.get_multiple_single_label_outputs(url)
        print('final output for {} : cat {} '.format(url,output))
#        print('final output for {} : cat {} {}'.format(url,output1,output2))
    elapsed_time = time.time()-start_time
    print('time per image:{}, {} elapsed for {} images'.format(elapsed_time/len(urls),elapsed_time,len(urls)))
#    cv2.imshow('output',output)

