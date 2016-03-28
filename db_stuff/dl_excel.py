import xlsxwriter
import re
from ..Yonti import drive
from .. import constants
from . import ebay_constants
db = constants.db


def fillTable(worksheet,main_categories,collection, archive, bold ,today):
    worksheet.write(0, 1, 'instock count', bold)
    worksheet.write(0, 2, collection.count(), bold)
    worksheet.write(0, 4, 'archive count', bold)
    worksheet.write(0, 5, archive.count(), bold)
    categories = []
    for cat in main_categories:
        print(cat)
        instock = collection.find({'categories': {'$eq': cat}}).count()
        new_items = collection.find({'$and': [{'categories': {'$eq': cat}},
                                              {'download_data.first_dl': {'$eq':today}}]}).count()
        archive = archive.find({'categories': {'$eq': cat}}).count()
        total = instock + archive
        categories.append([cat, instock, new_items, archive, total])
    categories_length =len(categories)+3
    worksheet.set_column('B:F',15)

    options = {'data' : categories,
               'total_row': True,
               'columns': [{'header': 'Categories','total_string': 'Total'},
                           {'header': 'instock'},
                           {'header': 'new instock items'},
                           {'header': 'archived'},
                           {'header': 'total'}]}

    worksheet.add_table('B2:F'+str(categories_length), options)
    for x in ['C', 'D', 'E', 'F']:
        worksheet.write_formula(x+str(categories_length), '=SUM('+x+'3:'+x+str(categories_length-1)+')')


def mongo2xl(collection_name, dl_info):
    if collection_name == 'ebay':
        filename = collection_name
    else:
        fail = False
        filename = 'empty'
        try:
            [filename, gender] = re.split("_", collection_name)
        except:
            fail = True
        if filename not in ["ShopStyle", "GangnamStyle"] or fail:
            print ('error in collection name')
            return

    path2file = '/home/developer/yonti/'+filename+'.xlsx'
    workbook = xlsxwriter.Workbook(path2file)
    bold = workbook.add_format({'bold': True})
    today = dl_info['date']
    worksheet_main = workbook.add_worksheet('main')
    worksheet_main.set_column('B:G',20)
    worksheet_main.write(0,1, "DOWNLOAD INFO")
    worksheet_main.write(1,1, "date")
    worksheet_main.write(1,2, today)
    worksheet_main.write(2,1, "duration")
    worksheet_main.write(2,2, dl_info['dl_duration'])
    worksheet_main.write(3,1, "instock items")
    worksheet_main.write(4, 1, "archived items")

    categories = list(set(ebay_constants.ebay_paperdoll_women.values()))
    categories.sort()

    instock_items = 0
    archived_items = 0
    if filename == 'ebay':

        for gender in ['Female', 'Male', 'Unisex','Tees']:
            print("working on "+ gender)
            collection = db['ebay_'+gender]
            archive = db['ebay_'+gender+'_archive']
            current_worksheet = workbook.add_worksheet(gender)
            fillTable(current_worksheet, categories, collection, archive, bold, today)
            instock_items += collection.count()
            archived_items += archive.count()

        store_info = dl_info["store_info"]
        try:
            s_i =[]
            for x in store_info:
                tmp_id = x["id"]
                try:
                    tmp_name = x["name"].decode("utf8")
                except:
                    tmp_name =""
                    print ("utf8 decode failed for %s" %str(tmp_id))
                tmp_item_count = x["items_downloaded"]
                tmp_duration = x["dl_duration"]
                tmp_link =  x["link"]
                tmp_modified = x["modified"]
                tmp_BW = x["B/W"]
                tmp_status = x["status"]
                tmp = [tmp_id,tmp_name,tmp_item_count,tmp_duration,tmp_link,tmp_modified,tmp_status,tmp_BW]
                print (tmp)
                s_i.append(tmp)

            for status in ["black","white"]:
                print("working on "+ status+"list")
                current_worksheet = workbook.add_worksheet(status+'list')

                dict2list = [y[:7] for y in s_i if y[7]==status]
                item_count = len(dict2list)

                current_worksheet.set_column('A:G',15)
                current_worksheet.set_column('B:B',25)
                current_worksheet.set_column('E:E',35)
                current_worksheet.set_column('F:F',25)
                current_worksheet.add_table('A1:G'+str(item_count+4),
                                            {'data': dict2list,
                                             'columns': [ {'header': 'id'},
                                                          {'header' : 'name'},
                                                          {'header' : 'items_downloaded'},
                                                          {'header' : 'download duration'},
                                                          {'header' : 'link'},
                                                          {'header': 'modified time'},
                                                          {'header': 'status'}],
                                             'banded_columns': True,
                                             'banded_rows': True})
        except:
            print ("error in blacklist/whitelist- saving to disk")
            f = open('/home/developer/yonti/tmp_store_info.log','w')
            for line in store_info:
                f.write(str(line.values())+'\n')
            f.close()

    else:
        for gender in ['Female', 'Male']:
            print("working on " + gender)
            collection = db[collection_name]
            arcive = db[collection_name+"_archive"]
            if gender is 'Female':
                current_worksheet = workbook.add_worksheet('Female')
            else :
                current_worksheet = workbook.add_worksheet('Male')

            fillTable(current_worksheet, categories, collection, arcive, bold, today)
            instock_items += collection.count()
            archived_items += archive.count()

    worksheet_main.write(3, 2, instock_items)
    worksheet_main.write(4, 2, archived_items)
    workbook.close()

    print ('uploading to drive...')
    files = (filename, path2file, True)
    res = drive.upload2drive(files)

    if res:
        print('file uploaded!')
    else:
        print ('error while uploading!')

