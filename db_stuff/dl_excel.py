import xlsxwriter
from ..Yonti import drive
from .. import constants
from . import ebay_constants
db = constants.db


def fillTable(worksheet,main_categories,collection,bold ,today):
    worksheet.write(0, 1, 'total count', bold)
    worksheet.write(0, 2, collection.count(), bold)
    categories = []
    for cat in main_categories:
        print(cat)
        items = collection.find({'categories': {'$eq': cat}}).count()
        new_items = collection.find({'$and': [{'categories': {'$eq': cat}}, {'download_data.first_dl': {'$eq':today}}]}).count()
        instock = collection.find({'$and': [{'categories': {'$eq': cat}}, {'status.instock': {'$eq': True}}]}).count()
        out = collection.find({'$and': [{'categories': {'$eq': cat}}, {'status.instock': {'$eq': False}}]}).count()
        categories.append([cat, items, new_items, instock, out])
    categories_length =len(categories)+3
    worksheet.set_column('B:F',15)

    options = {'data' : categories,
               'total_row': True,
               'columns': [{'header': 'Categories','total_string': 'Total'},
                           {'header': 'items'},
                           {'header': 'new_items'},
                           {'header': 'instock'},
                           {'header': 'out of stock'}]}

    worksheet.add_table('B2:F'+str(categories_length), options)
    for x in ['C', 'D', 'E', 'F']:
        worksheet.write_formula(x+str(categories_length), '=SUM('+x+'3:'+x+str(categories_length-1)+')')


def mongo2xl(filename, dl_info):
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
    worksheet_main.write(3,1, "total items")

    categories = list(set(ebay_constants.ebay_paperdoll_women.values()))
    categories.sort()

    total_items = 0
    if filename == 'ebay':

        for gender in ['Female', 'Male', 'Unisex','Tees']:
            print("working on "+ gender)
            if gender is 'Female':
                collection = db.ebay_Female
                current_worksheet = workbook.add_worksheet('Female')
            elif gender is 'Male':
                collection = db.ebay_Male
                current_worksheet = workbook.add_worksheet('Male')
            elif gender is 'Unisex':
                collection = db.ebay_Unisex
                current_worksheet = workbook.add_worksheet('Unisex')
            else:
                collection = db.ebay_Tees
                current_worksheet = workbook.add_worksheet('Tees')

            fillTable(current_worksheet, categories, collection, bold, today)
            total_items += collection.count()
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
        if filename == 'shopstyle':
            collection = db.products

        elif filename == 'flipkart':
            collection = db.flipkart

        elif filename == 'Gangnam':
            collection = db.Gangnam_Style_Female

        else:
            print ('nothing to convert')
            workbook.close()
            return

        total_items += collection.count()
        current_worksheet = workbook.add_worksheet('Women')
        fillTable(current_worksheet,categories,collection, bold,today)

    worksheet_main.write(3, 2, total_items)
    workbook.close()

    print ('uploading to drive...')
    files = (filename, path2file, True)
    res = drive.upload2drive(files)

    if res:
        print('file uploaded!')
    else:
        print ('error while uploading!')

