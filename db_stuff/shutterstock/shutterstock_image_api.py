from requests import get
from trendi.constants import db
from tqdm import tqdm
import json
from datetime import datetime
from time import sleep

client_credentials = {'client_id': '34c3fc04acd544e3cab1',
          'client_secret': 'e050e73728f3edcecaa7e3a2a18b9569aacf8c92'}

auth = (client_credentials['client_id'], client_credentials['client_secret'])

age_lookuptable = { 1 : 'infants',
                    2 : 'children',
                    3 : 'teenagers',
                    4 : '20s',
                    5 : '30s',
                    6 : '40s',
                    7 : '50s',
                    8 : '60s',
                    9 : 'older'}

ethnicity_lookuptable = { 1 : 'african',
                        2 : 'african_american',
                        3 : 'black',
                        4 : 'brazilian',
                        5 : 'chinese',
                        6 : 'caucasian',
                        7 : 'east_asian',
                        8 : 'hispanic',
                        9 : 'japanese',
                        10: 'middle_eastern',
                        11: 'native_american',
                        12: 'pacific_islander',
                        13: 'south_asian',
                        14: 'southeast_asian',
                        15: 'other'}

global_start = (2000, 1, 1)
today = datetime.today()
global_end = (today.year, today.month, today.day)

print('### SHUTTERSTOCK API IMAGE SCRAPER ###')
print('fill in the following fields:')
query_filter ={'query': raw_input('query by:'),
               'and_queries': [],
               'not_queries': []}

query_and_flag = True
while query_and_flag:
    tmp_and = raw_input('And by: (leave blank and enter to stop)')
    if len(tmp_and) < 2:
        query_and_flag = False
    else:
        query_filter['and_queries'].append(tmp_and)

query_not_flag = True
while query_not_flag:
    tmp_not = raw_input('and Not by: (leave blank and enter to stop)')
    if len(tmp_not) < 2:
        query_not_flag = False
    else:
        query_filter['not_queries'].append(tmp_not)

advanced = raw_input('do you want advanced people filters? (y/n) ') == 'y'
advanced_filter = {}
if advanced:
    gender = raw_input('filter by gender: (female/male/no) ')
    if gender in ['female', 'male']:
        advanced_filter['people_gender'] = gender

    age = raw_input('filter by age: (y/n)') == 'y'
    if age:
        print ('insert the age index:')
        for x in age_lookuptable:
            print('{}. {}'.format(x, age_lookuptable[x]))
        people_age = int(raw_input(''))
        if 0 < people_age < 10:
            advanced_filter['people_age'] = age_lookuptable[people_age]
            print ('you choose {}'.format(age_lookuptable[people_age]))

    ethnicity = raw_input('filter by people ethnicity? (y/n) ') == 'y'
    if ethnicity:
        print ('insert the ethnicity index:')
        for x in ethnicity_lookuptable:
            print('{}. {}'.format(x, ethnicity_lookuptable[x]))
        people_ethnicity = int(raw_input(''))
        if 0 < people_ethnicity < 16:
            advanced_filter['people_ethnicity'] = ethnicity_lookuptable[people_ethnicity]
            print ('you choose {}'.format(ethnicity_lookuptable[people_ethnicity]))

collection_name = raw_input('collection name: (either existing or new) ')
if collection_name not in db.collection_names():
    for key in ['id', 'category']:
        db[collection_name].create_index(key, background=True)


def build_date_string(date_tuple):
    date_string = '{}-{:0>2}-{:0>2}'.format(*date_tuple)
    return date_string


def req_wrapper(req):
    res = get(req, auth=auth)
    if res.status_code != 200:
        raise Warning('request denied!\nreq => {}'.format(req))

    res_dict = json.loads(res.text)
    if 'total_count' not in res_dict.keys():
        raise Warning('No total_count!\nreq => {}'.format(req))

    total_count = int(res_dict['total_count'])
    print res_dict
    return res_dict, total_count


def build_req_string(req_start, req_mid, start_date, end_date, per_page=1, page=1):
    start_str = build_date_string(start_date)
    end_str = build_date_string(end_date)
    dates_string = '&added_date_start={}&added_date_end={}'.format(start_str, end_str)
    pages = 'per_page={}&page={}'.format(per_page, page)
    req_final = req_start + pages + req_mid + dates_string
    return req_final


def divide_dates(pair):
    start_date, end_date = pair
    start_year, start_month, start_day = start_date
    end_year, end_month, end_day = end_date
    mid_date = ((end_year+start_year)/2, (end_month+start_month)/2, (end_day+start_day)/2)
    pair1 = (start_date, mid_date)
    if mid_date == start_date:
        if mid_date[1] < 12 and mid_date[1] < end_date[1]:
            mid_date =(mid_date[0], mid_date[1] + 1, mid_date[2])
        elif mid_date[2] < 28 and mid_date[2] < end_date[2]:
            mid_date = (mid_date[0], mid_date[1], mid_date[2] + 1)
        else:
            pair1 = ()

    pair2 = (mid_date, end_date)
    if mid_date == start_date:
        pair2 = ()

    return pair1, pair2


def build_date_pairs(req_start, req_mid, start_date, end_date):
    dates_list = []
    date_candidates = [(start_date, end_date)]

    while len(date_candidates):
        pair = date_candidates.pop()
        tmp_req = build_req_string(req_start, req_mid, *pair)
        if req_wrapper(tmp_req)[1] > 2000:
            dates_list.append(pair)
        else:
            pair1, pair2 = divide_dates(pair)
            if len(pair1):
                date_candidates.append(pair1)
            if len(pair2):
                date_candidates.append(pair2)

    return dates_list


def build_req_parts(main_query, advance_query):
    req_body = 'https://api.shutterstock.com/v2/images/search?'
    req_query = '&image_type=photo&query={}'.format(main_query['query'])
    for andQuery in main_query['and_queries']:
        q = '%20'.join(andQuery.split())
        req_query += '%20AND%20{}'.format(q)
    for notQuery in main_query['not_queries']:
        q = '%20'.join(notQuery.split())
        req_query += '%20NOT%20{}'.format(q)
    for k in advance_query.keys():
        req_query += '&{}={}'.format(k, advance_query[k])

    date_pairs = build_date_pairs(req_body, req_query, global_start, global_end)

    return req_body, req_query, date_pairs


reqBody, reqQuery, date_list = build_req_parts(query_filter, advanced_filter)
print (date_list)
category = query_filter['query']

for dp in tqdm(date_list):
    for page_number in range(1, 5):
        request = build_req_string(reqBody, reqQuery, *dp, per_page=500, page=page_number)
        response, count = req_wrapper(request)

        if 'data' not in response.keys():
            raise Warning('No data!\nreq => {}'.format(request))
        data = response['data']
        for item in data:
            idx = item['id']
            id_exists = db[collection_name].count({'id': idx})
            if id_exists:
                doc = {'id': item["id"],
                       'images': {'XLarge': item['assests']['preview']['url'],
                                  'Large': item['assests']['large_thuimb']['url'],
                                  'Small': item['assests']['small_thumb']['url'],
                                  'aspect_ratio': item['aspect']},
                       'description': item['description'],
                       'category': category,
                       'and': {x: i for x, i in enumerate(query_filter['and_queries'])},
                       'not': {y: j for y, j in enumerate(query_filter['not_queries'])},
                       'advanced': {k: advanced_filter[k] for k in advanced_filter.keys()}}
                db[collection_name].insert_one(doc)

        if page_number*500 > count:
            break

        sleep(1)









