import argparse
import re
from datetime import datetime

import xlsxwriter

from core import constants
from core.Yonti import drive

db = constants.db


def category_tree_status(worksheet, merge_format, bold):
    worksheet.write(0, 1, 'last update', bold)

    worksheet.merge_range('C1:E1', datetime.ctime(datetime.now()), merge_format)

    leafs = db.amazon_category_tree.find({'Children.count': 0})
    categories = []
    for leaf in leafs:
        name = leaf['Name']
        node_id = leaf['BrowseNodeId']
        parents = ''
        for par in leaf['Parents']:
            parents += par + ', '
        expected = leaf['TotalResultsExpected']
        downloaed = leaf['TotalDownloaded']
        last_price = leaf['LastPrice']
        status = leaf['Status']
        categories.append([name, node_id, status, last_price, parents, expected, downloaed])
    categories_length = leafs.count()+3
    worksheet.set_column('B:H', 20)
    worksheet.set_column('F:F', 50)
    options = {'data': categories,
               'total_row': True,
               'columns': [{'header': 'Leaf', 'total_string': 'Total'},
                           {'header': 'node id'},
                           {'header': 'status'},
                           {'header': 'last price'},
                           {'header': 'parents'},
                           {'header': 'expected'},
                           {'header': 'downloaded'}]}

    worksheet.add_table('B2:H'+str(categories_length), options)
    for x in ['G', 'H']:
        worksheet.write_formula(x+str(categories_length), '=SUM('+x+'3:'+x+str(categories_length-1)+')')


def fill_table(worksheet, main_categories, collection, archive, bold, today):
    worksheet.write(0, 1, 'instock count', bold)
    worksheet.write(0, 2, collection.count(), bold)
    worksheet.write(0, 4, 'archive count', bold)
    worksheet.write(0, 5, archive.count(), bold)
    categories = []
    for cat in main_categories:
        print(cat)
        instock = collection.find({'categories': {'$eq': cat}}).count()
        new_items = collection.find({'$and': [{'categories': {'$eq': cat}},
                                              {'download_data.first_dl': {'$eq': today}}]}).count()
        archived = archive.find({'categories': {'$eq': cat}}).count()
        total = instock + archived
        categories.append([cat, instock, new_items, archived, total])
    categories_length = len(categories)+3
    worksheet.set_column('B:F', 15)

    options = {'data' : categories,
               'total_row': True,
               'columns': [{'header': 'Categories', 'total_string': 'Total'},
                           {'header': 'instock'},
                           {'header': 'new instock items'},
                           {'header': 'archived'},
                           {'header': 'total'}]}

    worksheet.add_table('B2:F'+str(categories_length), options)
    for x in ['C', 'D', 'E', 'F']:
        worksheet.write_formula(x+str(categories_length), '=SUM('+x+'3:'+x+str(categories_length-1)+')')


def mongo2xl(collection_name, dl_info):
    # if collection_name == 'ebay':
    #     filename = collection_name
    # else:
    #     fail = False
    #     filename = 'empty'

    try:
        filename = re.split("_", collection_name)[0]
    except StandardError as e:
        print (e)
        return

    path2file = '/home/developer/yonti/'+filename+'.xlsx'
    workbook = xlsxwriter.Workbook(path2file)
    bold = workbook.add_format({'bold': True})
    today = dl_info['start_date']
    worksheet_main = workbook.add_worksheet('main')
    worksheet_main.set_column('B:G', 25)
    worksheet_main.write(0, 1, "DOWNLOAD INFO")
    worksheet_main.write(1, 1, "start_date")
    worksheet_main.write(1, 2, today)
    worksheet_main.write(2, 1, "duration")
    worksheet_main.write(2, 2, dl_info['dl_duration'])
    worksheet_main.write(3, 1, "instock items")
    worksheet_main.write(4, 1, "archived items")

    worksheet_main.write(6, 1, "collection")
    worksheet_main.write(6, 2, collection_name)
    worksheet_main.write(7, 1, "items before")
    worksheet_main.write(7, 2, dl_info['items_before'])
    worksheet_main.write(8, 1, "items after")
    worksheet_main.write(8, 2, dl_info['items_after'])
    worksheet_main.write(9, 1, "items downloaded today")
    worksheet_main.write(9, 2, dl_info['items_new'])

    instock_items = 0
    archived_items = 0
    if filename == 'ebay':
        pass
        # from . import ebay_constants
        # categories_female = categories_male = list(set(ebay_constants.ebay_paperdoll_women.values()))
        # categories.sort()
        # for gender in ['Female', 'Male', 'Unisex']:#,'Tees']:
        #     print("working on "+ gender)
        #     collection = db['ebay_'+gender]
        #     archive = db['ebay_'+gender+'_archive']
        #     current_worksheet = workbook.add_worksheet(gender)
        #     fill_table(current_worksheet, categories, collection, archive, bold, today)
        #     instock_items += collection.count()
        #     archived_items += archive.count()
        #
        # store_info = dl_info["store_info"]
        # try:
        #     s_i =[]
        #     for x in store_info:
        #         tmp_id = x["id"]
        #         try:
        #             tmp_name = x["name"].decode("utf8")
        #         except:
        #             tmp_name =""
        #             print ("utf8 decode failed for %s" %str(tmp_id))
        #         tmp_item_count = x["items_downloaded"]
        #         tmp_duration = x["dl_duration"]
        #         tmp_link =  x["link"]
        #         tmp_modified = x["modified"]
        #         tmp_BW = x["B/W"]
        #         tmp_status = x["status"]
        #         tmp = [tmp_id,tmp_name,tmp_item_count,tmp_duration,tmp_link,tmp_modified,tmp_status,tmp_BW]
        #         print (tmp)
        #         s_i.append(tmp)
        #
        #     for status in ["black","white"]:
        #         print("working on "+ status+"list")
        #         current_worksheet = workbook.add_worksheet(status+'list')
        #
        #         dict2list = [y[:7] for y in s_i if y[7]==status]
        #         item_count = len(dict2list)
        #
        #         current_worksheet.set_column('A:G',15)
        #         current_worksheet.set_column('B:B',25)
        #         current_worksheet.set_column('E:E',35)
        #         current_worksheet.set_column('F:F',25)
        #         current_worksheet.add_table('A1:G'+str(item_count+4),
        #                                     {'data': dict2list,
        #                                      'columns': [ {'header': 'id'},
        #                                                   {'header' : 'name'},
        #                                                   {'header' : 'items_downloaded'},
        #                                                   {'header' : 'download duration'},
        #                                                   {'header' : 'link'},
        #                                                   {'header': 'modified time'},
        #                                                   {'header': 'status'}],
        #                                      'banded_columns': True,
        #                                      'banded_rows': True})
        # except:
        #     print ("error in blacklist/whitelist- saving to disk")
        #     f = open('/home/developer/yonti/tmp_store_info.log','w')
        #     for line in store_info:
        #         f.write(str(line.values())+'\n')
        #     f.close()
    # else:
    elif filename == 'recruit':
        from core.db_stuff.recruit.recruit_constants import recruit2category_idx
        categories_female = categories_male = list(set(recruit2category_idx.keys()))
    elif filename == 'amazon' or filename == 'amaze':
        from core.db_stuff.amazon.amazon_constants import amazon_categories_list
        categories_female = categories_male = amazon_categories_list
    else:
        from core.db_stuff.shopstyle.shopstyle_constants import shopstyle_paperdoll_female, shopstyle_paperdoll_male
        categories_female = list(set(shopstyle_paperdoll_female.values()))
        categories_male = list(set(shopstyle_paperdoll_male.values()))

    categories_female.sort()
    categories_male.sort()

    for col_gender in ['Female', 'Male']:
        tmp = filename + "_" + col_gender
        if filename == 'ebay':
            tmp += '_US'
        if filename == 'amazon':
            tmp = 'amazon_US_' + col_gender
        print("working on " + tmp)
        collection = db[tmp]
        archive = db[tmp+"_archive"]
        if col_gender is 'Female':
            categories = categories_female
            current_worksheet = workbook.add_worksheet('Female')
        else:
            categories = categories_male
            current_worksheet = workbook.add_worksheet('Male')

        fill_table(current_worksheet, categories, collection, archive, bold, today)
        instock_items += collection.count()
        archived_items += archive.count()

    worksheet_main.write(3, 2, instock_items)
    worksheet_main.write(4, 2, archived_items)
    if filename == 'amazon':
        merge_format = workbook.add_format({'align': 'center'})
        current_worksheet = workbook.add_worksheet('categories_tree')
        category_tree_status(current_worksheet, merge_format, bold)
    workbook.close()

    print ('uploading to drive...')
    files = (filename, path2file, True)
    res = drive.upload2drive(files)

    if res:
        print('file uploaded!')
    else:
        print ('error while uploading!')

    return


def get_user_input():
    parser = argparse.ArgumentParser(description='"@@@ create excel and upload to drive @@@')
    parser.add_argument('-n', '--name', default="ShopStyle", dest="name",
                        help='collection name - currently only ShopStyle, GangnamStyle, amaze or amazon')
    parser.add_argument('-g', '--gender', dest="gender",
                        help='specify which gender to download. (Female or Male - case sensitive)', required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    import sys
    import datetime
    current_dl_date = str(datetime.datetime.date(datetime.datetime.now()))

    user_input = get_user_input()
    col = user_input.name
    gender = user_input.gender

    if gender in ['Female', 'Male'] and col in ["ShopStyle", "GangnamStyle", "amaze"]:
        col = col + "_" + gender
    elif gender in ['Female', 'Male'] and col == 'amazon':
        col = col + "_US_" + gender
    else:
        print("bad input - gender should be only Female or Male (case sensitive)")
        sys.exit(1)

    info = {"start_date": current_dl_date,
            "dl_duration": 0,
            "items_before": 0,
            "items_after": 0,
            "items_new": 0}

    mongo2xl(col, info)

    print (col + "Update Finished!!!")
    sys.exit(0)
