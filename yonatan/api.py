#!/usr/bin/env python

import urllib2
import json
import facebook

locu_api = '17650e070e2bfccab3dd8f9605583793622ea6c1'


def locu_search(query):
    api_key = locu_api
    url = 'https://api.locu.com/v1_0/venue/search/?api_key=' + api_key
    locality = query.replace(' ', '%20')
    final_url = url + "&locality=" + locality + "&category=restaurant"
    json_obj = urllib2.urlopen(final_url)
    data = json.load(json_obj)
    for item in data['objects']:
        print item['name'], item['phone']


def fullContactCollect(email):
    api_key = '3558403aafc12f64'
    email = email
    fullURL = 'http://api.fullcontact.com/v2/person.json?apiKey=' + api_key + '&email=' + email
    loadUrl = urllib2.urlopen(fullURL)
    json_data = json.load(loadUrl)

    # print json_data
    # photos = json_data['photos']

    for item in json_data['socialProfiles']:
        if 'followers' in item:
            print str(item['typeName']) + ' - followers: ' + str(item['followers'])


# fullContactCollect('markcuban@dallasmavs.com')

my_access_token = 'EAACEdEose0cBAAz0Ova3dDgTF6PsBXk8tKdduBmoCnm71zsI14V77dBFnlykuZCa5AmA3vaubZCaDtZBZByPwUfkscsYU0GB2rvSRzZCtRTzq46A8GSuGHgZCjnDFJoVSKnP8B8jKeo50DRqmwVh5fzyikPBH4kJwicMdIZAc3JngZDZD'

my_fb_user_id = '693821542'


your_access_token = my_access_token

def get_fb_page_like_count(fb_page_id):
    graph = facebook.GraphAPI(access_token=your_access_token)
    args = {'fields': 'likes'}
    page = graph.get_object(fb_page_id, **args)
    print page.get('likes', 0)
    return page.get('likes', 0)

get_fb_page_like_count(my_fb_user_id)
