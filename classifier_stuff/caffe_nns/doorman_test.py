__author__ = 'jeremy'
import subprocess

def get_doorman_test_image_names():
    command = 'gsutil'
    bucket = 'gs://tg-training/doorman/relevant'
    retval = subprocess.call([command, 'ls ' + bucket])
