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

my_access_token = 'EAACEdEose0cBAB3PI5trgsPD59BbayD8gdbCDjaTzBE2f6vEcw3TAq97X21zOcnK8ZBb6XzLNk6S19JfNCyZCw4j1aJbpGX8vG5TirwPlBtXsRCYw95GVMAdHtIR1VR2tEK3LhOfKTI9XXsWgneq2uZA5q5Tk9n5iE1Qly9iwZDZD'
michal_access_token = 'EAACEdEose0cBAAOu0VAmvxRKMedjuAKV8ZAkaX4PByAZA2D327qjpKj8XsILkYRKdiiWqWg2HkPZB4JvqR8jiyGtprH4Bs6IdHLzC5bSotydXJnNBnJr42MaW95vDdaORW8b3jx7WNeoEImyNBjd71VINNUQUI9UE5z23cWugZDZD'
Hilush_access_token = 'EAACEdEose0cBAMFjZCQ2dg2p7KiZA7K6NYhBQQ6DSIK4k0sFBHGJTZBeC6V0OZA61KuUKCOGsBmIcZCfsiGVvJ1b8Y47uNbZCtZAUQTygw7JitGyhPZBZCiOWxkXjzijlyRxJRwRCX1HohX2cGL2RpxN4KOdqvpEhaDIDWHfvud6AQgZDZD'
technoart_org_token = 'EAACEdEose0cBAOr1SsAZCBRBGcZAp3yST7ZAhE9FaJzEvNSfZAR45vDapHTdTtZBd1ivDZCNLrqUesTuZCEuURwORZCDJC1ZAEfi4ZANu9apoBRFZBuOekoRb3r191fA4xxy68gH6OCBJ5K0fhwiHNKYzWcPB3N46eiUIvGeBfZCqmOsycPZBVK8GEK6xLVRgZAN4a7ulRK7hEhZBwo8wZDZD'

my_fb_user_id = '693821542'
michal_fb_page_id = '423546661173003'
Hilush_fb_user_id = '1043151871'
technoart_org_id = '1659879187584164'

profile_pic_id = "432754976542"
paris_album_id = '10153673942246543'

your_access_token = my_access_token

def get_fb_page_like_count(my_fb_user_id):
    graph = facebook.GraphAPI(access_token=technoart_org_token)
    # args = {'fields': 'photos'}
    # page = graph.get_object(id=my_fb_user_id, **args)
    # facebook_user = facebook.GraphAPI(my_access_token).get_object('me')
    # page = graph.get_object(paris_album_id)
    # post = graph.get_object(id=my_fb_user_id)
# page insights of people who liked the page #
    page_fans_gender_age = graph.get_connections(id=my_fb_user_id, connection_name='insights/page_fans_gender_age')
    page_fans_country = graph.get_connections(id=my_fb_user_id, connection_name='insights/page_fans_country')
    page_fans_city = graph.get_connections(id=my_fb_user_id, connection_name='insights/page_fans_city')
    page_fans_locale = graph.get_connections(id=my_fb_user_id, connection_name='insights/page_fans_locale')

# page insights of people who saw the page #
    page_impressions_by_age_gender_unique = graph.get_connections(id=my_fb_user_id, connection_name='insights/page_impressions_by_age_gender_unique/days_28')
    page_impressions_by_country_unique = graph.get_connections(id=my_fb_user_id, connection_name='insights/page_impressions_by_country_unique/days_28')
    page_impressions_by_locale_unique = graph.get_connections(id=my_fb_user_id, connection_name='insights/page_impressions_by_locale_unique/days_28')
    page_impressions_by_city_unique = graph.get_connections(id=my_fb_user_id, connection_name='insights/page_impressions_by_city_unique/days_28')
    # friends = graph.get_connections(id=my_fb_user_id)

    tyota = graph.get_connections(id=my_fb_user_id, connection_name='insights')
    #print tyota
    #for key, value in tyota.iteritems():
    #    print "{0} : {1}\n".format(key, value)


    sum = 0
    print page_fans_country['data'][0]['description']
    for key, value in page_fans_country['data'][0]['values'][2]['value'].iteritems():
        print "{0} : {1}".format(key, value)

    print page_fans_city['data'][0]['description']
    for key, value in page_fans_city['data'][0]['values'][2]['value'].iteritems():
        print "{0} : {1}".format(key.encode('ascii','ignore'), value)

    print page_fans_locale['data'][0]['description']
    for key, value in page_fans_locale['data'][0]['values'][2]['value'].iteritems():
        print "{0} : {1}".format(key, value)

    print page_fans_gender_age['data'][0]['description']
    for key, value in page_fans_gender_age['data'][0]['values'][2]['value'].iteritems():
        print "{0} : {1}".format(key, value)
        sum += value
    print "sum : {0}\n\n".format(sum)


    print page_impressions_by_country_unique['data'][0]['description']
    for key, value in page_impressions_by_country_unique['data'][0]['values'][2]['value'].iteritems():
        print "{0} : {1}".format(key, value)

    print page_impressions_by_city_unique['data'][0]['description']
    for key, value in page_impressions_by_city_unique['data'][0]['values'][2]['value'].iteritems():
        print "{0} : {1}".format(key.encode('ascii','ignore'), value)

    print page_impressions_by_locale_unique['data'][0]['description']
    for key, value in page_impressions_by_locale_unique['data'][0]['values'][2]['value'].iteritems():
        print "{0} : {1}".format(key, value)

    print page_impressions_by_age_gender_unique['data'][0]['description']
    for key, value in page_impressions_by_age_gender_unique['data'][0]['values'][2]['value'].iteritems():
        print "{0} : {1}".format(key, value)

    # print page['comments']
    # return page.get
    # print facebook_user


get_fb_page_like_count(technoart_org_id)
