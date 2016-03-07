import xlsxwriter
from .Yonti import drive
from . import constants
from . import ebay_constants
import datetime

today_date = str(datetime.datetime.date(datetime.datetime.now()))

db = constants.db

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
    worksheet_main.write(2,4, dl_info['blacklist'])
    worksheet_main.write(1,6, "whitelist")
    worksheet_main.write(2,6, dl_info['whitelist'])


    ppd_categories = list(set(ebay_constants.ebay_paperdoll_women.values()))
    for gender in ['Female']:#'Male','Unisex']:
        if gender is 'Female':
            collection = db.ebay_Female
            current_worksheet = workbook.add_worksheet('Female')
        elif gender is 'Male':
            collection = db.ebay_Male
            current_worksheet = workbook.add_worksheet('Male')
        else:
            collection = db.ebay_Unisex
            current_worksheet = workbook.add_worksheet('Unisex')

        current_worksheet.write(0, 1, 'total count', bold)
        current_worksheet.write(0, 2, collection.count(), bold)
        categories = []
        for cat in ppd_categories:
            items = collection.find({'categories': cat}).count()
            new_items = collection.find({'categories': cat, 'download_data.first_dl': today_date}).count()
            instock = collection.find({'categories': cat, 'status.instock': True}).count()
            out = collection.find({'categories': cat, 'status.instock': False}).count()
            categories.append([cat, items,new_items, instock, out])

        current_worksheet.set_column('B:F',15)
        current_worksheet.add_table('B2:F'+str(len(categories)+4),
                                 {'data' : categories,
                                  'header_row': True,
                                  'autofilter': True,
                                  'total_row': True,
                                  'columns': [{'header': 'Category', 'total_string': 'Total'},
                                              {'header': 'items',    'total_function' : 'sum'},
                                              {'header': 'new items','total_function' : 'sum'},
                                              {'header': 'instock',  'total_function' : 'sum'},
                                              {'header': 'out of stock', 'total_function' : 'sum'}],
                                  'banded_columns': True,
                                  'banded_rows': False})
    # for item, cost in (categories):
    #     worksheet_main.write(row, col,     item)
    #     worksheet_main.write(row, col + 1, cost)
    #     row += 1

    # Write a total using a formula.
    # lastrow = row-1
    # worksheet_main.write(row, 0,                      'Total', bold)
    # worksheet_main.write(row, 1, '=SUM(B1:B'+str(lastrow)+')', bold)

    workbook.close()

    print ('uploading to drive...')
    files = (filename,path2file, True)
    res = drive.upload2drive(files)
    if res:
        print('file uploaded!')
    else:
        print ('error while uploading!')