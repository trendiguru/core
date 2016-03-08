import xlsxwriter
from .Yonti import drive
from . import constants
from . import ebay_constants
import datetime

today_date = str(datetime.datetime.date(datetime.datetime.now()))

db = constants.db
def fillTable(worksheet,main_categories,collection,bold):
    worksheet.write(0, 1, 'total count', bold)
    worksheet.write(0, 2, collection.count(), bold)
    categories = []
    for cat in main_categories:
        items = collection.find({'categories': cat}).count()
        new_items = collection.find({'categories': cat, 'download_data.first_dl': today_date}).count()
        instock = collection.find({'categories': cat, 'status.instock': True}).count()
        out = collection.find({'categories': cat, 'status.instock': False}).count()
        categories.append([cat, items,new_items, instock, out])
    categories_length =len(categories)+2
    worksheet.set_column('B:F',15)
    worksheet.add_table('B2:F'+str(categories_length+1),
                     {'data' : categories,
                      'total_row': True,
                      'columns': [{'header': 'Category', 'total_string': 'Total'},
                                  {'total_function' : 'sum','header': 'items'},
                                  {'total_function' : 'sum','header': 'new items'},
                                  { 'total_function' : 'sum','header': 'instock'},
                                  { 'total_function' : 'sum'}],})#'header': 'out of stock',


def mongo2xl(filename, dl_info):
    path2file = '/home/developer/yonti/'+filename+'.xlsx'
    workbook = xlsxwriter.Workbook(path2file)
    bold = workbook.add_format({'bold': True})
    today_date = dl_info['date']
    worksheet_main = workbook.add_worksheet('main')
    worksheet_main.set_column('B:G',15)
    worksheet_main.write(1,1, "date")
    worksheet_main.write(2,1, today_date)
    worksheet_main.write(1,2, "duration")
    worksheet_main.write(2,2, dl_info['dl_duration'])
    worksheet_main.write(1,4, "blacklist")
    for i,b in enumerate(dl_info['blacklist']):
        worksheet_main.write(2+i,4, b)
    worksheet_main.write(1,6, "whitelist")
    for i,w in enumerate(dl_info['whitelist']):
        worksheet_main.write(2+i,6, w)

    categories = list(set(ebay_constants.ebay_paperdoll_women.values()))

    if filename == 'ebay':
        for gender in ['Female', 'Male', 'Unisex']:
            if gender is 'Female':
                collection = db.ebay_Female
                current_worksheet = workbook.add_worksheet('Female')
            elif gender is 'Male':
                collection = db.ebay_Male
                current_worksheet = workbook.add_worksheet('Male')
            else:
                collection = db.ebay_Unisex
                current_worksheet = workbook.add_worksheet('Unisex')

            fillTable(current_worksheet, categories, collection, bold)

        current_worksheet = workbook.add_worksheet('ftp folder')
        current_worksheet.set_column('A:D',25)
        current_worksheet.add_table('A1:D'+str(len(dl_info['raw_data'])+4),
                                    {'data': dl_info['raw_data'],
                                     'columns': [ {'header': 'Filename'},
                                                  {'header': 'Update time'},
                                                  {'header': 'Size'},
                                                  {'header': 'Status'}],
                                     'banded_columns': True,
                                     'banded_rows': True})
    else:
        if filename == 'shopstyle':
            collection = db.products

        elif filename == 'flipkart':
            collection = db.flipkart

        else:
            print ('nothing to convert')
            workbook.close()
            return
        current_worksheet = workbook.add_worksheet('Women')
        fillTable(current_worksheet,categories,collection, bold)

    workbook.close()

    print ('uploading to drive...')
    files = (filename, path2file, True)
    res = drive.upload2drive(files)
    if res:
        print('file uploaded!')
    else:
        print ('error while uploading!')

