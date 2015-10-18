__author__ = 'yonatan'

"""
1. run getShopStyleDB.py
2. do stats
3. mail the output to relevent
"""

import json

import constants

db = constants.db


def download_stats():
    dl_data = db.download_data.find()[0]
    date = dl_data['current_dl']
    stats = {'date': date,
             'items_downloaded': dl_data['items_downloaded'],
             'existing_items': dl_data['existing_items'],
             'new_items': dl_data['new_items'],
             'items_from_archive': dl_data['returned_from_archive'],
             'items_sent_to_archive': dl_data['sent_to_archive'],
             'dl_duration(hours)': dl_data['total_dl_time(hours)'],
             'items_by_category': {}}
    for i in constants.db_relevant_items:
        if i == 'women' or i == 'women-cloth':
            continue
        stats['items_by_category'][i] = {'total': db.products.find({'categories.id': i}).count(),
                                         'new': db.products.find({'$and': [{'categories.id': i},
                                                                           {'download_data.first_dl': date}]}).count()}
    with open(date + '.txt', 'w') as outfile:
        json.dump(stats, outfile)

    print stats


if __name__ == "__main__":
    # update_db = getShopStyleDB.ShopStyleDownloader()
    # update_db.run_by_category(type="DAILY")
    download_stats()
    # e_mail
