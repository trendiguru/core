__author__ = 'jeremy'

from trendi.paperdoll import pd_falcon_client

def get_pd_results():
    url = 'https://thechive.files.wordpress.com/2017/02/0c7bf9a4951ade636082e45849b01cd8.jpeg'
    resp = pd_falcon_client.pd(url)
    print('resp:'+str(resp))