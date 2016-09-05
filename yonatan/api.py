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

my_access_token = 'EAACEdEose0cBAHLj2S88B0HvAakO7UZCUmRF9Bou7agidrSCZA9R4glbq0w8PzvZAnLZBHdMWeYoCMRoEno70LSq9FQWvQsHZBxkQfnXcagyKkWFSEP7CuTbXZABHwIZAQAT3p98NxazAO95KZAKE2AcvXZCpm8ZBHY4ENKbWkIWrDdDT6J45asZC60iWb6xnZAxcLz70uW3t7ZCaAAZDZD'
michal_access_token = 'EAACEdEose0cBAOhZAAPZAhChSgjRwdHsbS8oTO6xvXSatkciSYQiZCZCTdlFmJAPfwjR3gjcThzmhkZC75wwbkkAqTbNUlMKs60Dx18uFpXJbgKZATRI5oSmQyCgHYQqF8X1lALQimDwNich5cLsXRJM6gAJDel5AtcZB8r58FHjgZDZD'
Hilush_access_token = 'EAACEdEose0cBAMFjZCQ2dg2p7KiZA7K6NYhBQQ6DSIK4k0sFBHGJTZBeC6V0OZA61KuUKCOGsBmIcZCfsiGVvJ1b8Y47uNbZCtZAUQTygw7JitGyhPZBZCiOWxkXjzijlyRxJRwRCX1HohX2cGL2RpxN4KOdqvpEhaDIDWHfvud6AQgZDZD'

my_fb_user_id = '693821542'
michal_fb_page_id = '423546661173003'
Hilush_fb_user_id = '1043151871'

your_access_token = my_access_token

def get_fb_page_like_count(my_fb_user_id):
    graph = facebook.GraphAPI(access_token=my_access_token)
    args = {'fields': 'photos'}
    page = graph.get_object(my_fb_user_id, **args)

    # facebook_user = facebook.GraphAPI(my_access_token).get_object('me')
    # page = graph.get_object(my_fb_user_id)
    # post = graph.get_object(id=my_fb_user_id)
    friends = graph.get_connections(id=my_fb_user_id, connection_name='fans')
    # friends = graph.get_connections(id=my_fb_user_id)

    # print friends

    print page
    return page.get
    # print facebook_user

get_fb_page_like_count(my_fb_user_id)
