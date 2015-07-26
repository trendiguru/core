__author__ = 'jeremy'
# paperdoll run from matlab
# this needs to run on server that
# has matlab running w. opencv3
#which is currently 'mightili.trendi.guru'


import subprocess
import shutil
import requests

import matlab.engine


def get_parse_from_matlab(image_filename):
    eng = matlab.engine.start_matlab('-nodesktop')
    result = eng.pd("inputimg.jpg")
    stripped_name = image_filename.split('.jpg')[0]
    outfilename = stripped_name + '.png'
    print('outfilename:' + outfilename)
    subprocess.Popen("scp -i /home/jeremy/first_aws.pem  output.png ubuntu@extremeli.trendi.guru:" + outfilename,
                     shell=True, stdout=subprocess.PIPE).stdout.read()
    subprocess.Popen("cp inputimg.jpg " + outfilename, shell=True, stdout=subprocess.PIPE).stdout.read()
    #scp -i ~/first_aws.pem  img.jpg ubuntu@extremeli.trendi.guru:.
    return result


def get_parse_mask(image_url):
    # try:
    # print("trying remotely (url) ")
    #       testfile = urllib.URLopener()
    #    except:
    #        print("error in urlopener"+str(sys.exc_info()[0]))
    #        return None
    #   try:#
    ##      infilename='/home/jeremy/infile.jpg'
    #        print("url:"+str(image_url)+" infile:"+str(infilename))#
    #       testfile.retrieve(image_url,infilename)
    #   except:
    #       print('err in retreive'+str(sys.exc_info()[0]))

    response = requests.get(image_url, stream=True)
    with open('inputimg.jpg', 'wb') as out_file:
        shutil.copyfileobj(response.raw, out_file)
    del response

    # img_array = imdecode(np.asarray(bytearray(response.content)), 1)
    stripped_name = image_url.split('//')[1]
    modified_name = stripped_name.replace('/', '_')
    print('stripped name:' + stripped_name)
    print('modified name:' + modified_name)
    #        cv2.imwrite(img_array,stripped_name)
    result = get_parse_from_matlab(modified_name)
    return result
