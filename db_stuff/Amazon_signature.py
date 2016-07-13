import base64
import hmac
import hashlib
import re
import time


def sort_by_ascii_value(left, right):
    if not len(left) and not len(right):
        return 0
    elif not len(left) and len(right):
        return -1
    elif not len(right) and len(left):
        return 1
    else:
        lefty = ord(left[0])
        righty = ord(right[0])
        if lefty < righty:
            return -1
        elif lefty > righty:
            return 1
        else:
            return sort_by_ascii_value(left[1:],right[1:])


def encode_me(string_2_encode):
    for pair in [(r'/', '%2F'), (r'\+', '%2B'), (r'=', '%3D'), (r'\:', '%3A'), (r',', '%2C'), (r' ', '%20')]:
        split_slash = re.split(pair[0],string_2_encode)
        string_2_encode = split_slash[0]
        for i in range(1, len(split_slash)):
            string_2_encode += pair[1]+split_slash[i]

    return string_2_encode


def create_signature(parameters, aws_secret_access_key, get_or_post):
    """
    1. rewrites the parameters dict as a string
    2. encodes whatever is needed
    3. creates encoded signature
    """
    keys = parameters.keys()
    keys.sort(sort_by_ascii_value)
    message = get_or_post +'\n'+'webservices.amazon.com'+'\n'+'/onca/xml'+'\n'
    params=''
    for key in keys:
        value = parameters[key]
        if key in ['Timestamp','ResponseGroup','Keywords','Title']:
            value = encode_me(value)
        params += "%s=%s" % (key, value)+'&'
    message += params[:-1]  # to remove unwanted & sign in the end of the string
    signature_raw = base64.encodestring(hmac.new(aws_secret_access_key, message, hashlib.sha256).digest()).strip()
    signature_encoded = encode_me(signature_raw)

    return params, signature_encoded


def get_amazon_signed_url(parameters, get_or_post='GET', print_flag=True):
    """
    example parameters:
        aws_access_key = 'your_aws_access_key'  # DONT FORGET
        associate_tag = 'your_associate_tag'  # DONT FORGET
        parameters = {
                 'AWSAccessKeyId': aws_access_key,
                 'AssociateTag':associate_tag,
                 'Availability':'Available',
                 'Brand':'Lacoste',
                 'Keywords':'shirts',
                 'Operation':'ItemSearch',
                 'SearchIndex':'FashionWomen',
                 'Service':'AWSECommerceService',
                 'Timestamp': time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 'ResponseGroup':'ItemAttributes,Offers,Images,Reviews'}
    note that the following keys are mandatory:
    'AWSAccessKeyId'
    'AssociateTag'
    'Operation'
    'Timestamp'
    """

    aws_secret_access_key = 'r82svvj4F8h6haYZd3jU+3HkChrW3j8RGcW7WXRK'
    parameters, signature = create_signature(parameters, aws_secret_access_key, get_or_post)
    url = 'http://webservices.amazon.com/onca/xml?'
    amazon_signed_url = url+parameters+'Signature='+signature
    if print_flag:
        print(amazon_signed_url)
    return amazon_signed_url

#
# parameters = {
#     'AWSAccessKeyId': 'AKIAIQJZVKJKJUUC4ETA',
#     'AssociateTag': 'fazz0b-20',
#     'Availability': 'Available',
#     # 'Title': 'mini dress',
#     # 'Operation': 'ItemLookup',
#     'Operation': 'ItemSearch',
#     'SearchIndex': 'FashionWomen',
#     'Service': 'AWSECommerceService',
#     'Timestamp': time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
#     # 'IdType': 'ASIN',
#     # 'ItemId': 'B01CT51MXC',
#     # 'MaximumPrice':'3100',
#     # 'MinimumPrice':'3099',
#     'ResponseGroup': 'ItemAttributes, OfferSummary,Images'}
#
# parameters['Operation'] = 'BrowseNodeLookup'
# # parameters['ResponseGroup'] = 'SearchBins'# 'BrowseNodeInfo'
# # parameters['BrowseNode'] = '7147443011'
# parameters['ResponseGroup'] = 'BrowseNodeInfo'
# parameters['BrowseNodeId'] = '7141124011'
# get_amazon_signed_url(parameters, 'GET')

