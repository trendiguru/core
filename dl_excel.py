import xlsxwriter
from .Yonti import drive
from . import constants
from . import ebay_constants
db = constants.db


def fillTable(worksheet,main_categories,collection,bold ,today):
    worksheet.write(0, 1, 'total count', bold)
    worksheet.write(0, 2, collection.count(), bold)
    categories = []
    for cat in main_categories:
        items = collection.find({'categories': cat}).count()
        new_items = collection.find({'categories': cat, 'download_data.first_dl': today}).count()
        instock = collection.find({'categories': cat, 'status.instock': True}).count()
        out = collection.find({'categories': cat, 'status.instock': False}).count()
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

        for status in ["black","white"]:
            current_worksheet = workbook.add_worksheet(status+'list')

            dict2list = [[x["id"],x["name"],x["items_downloaded"],x["dl_duration"],x["link"],x["modified"]]
                         for x in dl_info["store_info"] if x["B/W"] == status]
            item_count = len(dict2list)

            current_worksheet.set_column('A:G',15)
            current_worksheet.add_table('A1:G'+str(item_count+4),
                                        {'data': dl_info['raw_data'],
                                         'columns': [ {'header': 'id'},
                                                      {'header' : 'name'},
                                                      {'header' : 'items_downloaded'},
                                                      {'header' : 'download duration'},
                                                      {'header' : 'link'},
                                                      {'header': 'modified time'}],
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

