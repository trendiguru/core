__author__ = 'jeremy'
# paperdoll run from matlab
# this needs to run on server that
# has matlab running w. opencv3
#which is currently 'mightili.trendi.guru'
import matlab.engine

def get_parse_mask(image_filename):
    eng = matlab.engine.start_matlab()
    result = eng.pd(image_filename)
    # scp -o UserKnownHostsFile=/dev/null -o CheckHostIP=no -o StrictHostKeyChecking=no -i .ssh/google_compute_engine jeremy@mightili.trendi.guru:/home/jeremy/img.jpg .
    # scp -i ~/first_aws.pem  img.jpg ubuntu@extremeli.trendi.guru:.

    return result


if __name__ == "__main__":
    get_parse_mask('img.jpg')
