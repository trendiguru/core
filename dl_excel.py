import xlsxwriter
from .Yonti import drive
from . import constants
from . import ebay_constants
import datetime

today_date = str(datetime.datetime.date(datetime.datetime.now()))

db = constants.db

filename = 'ebay'
path2file = '/home/developer/yonti/ebay.xlsx'
workbook = xlsxwriter.Workbook(path2file)
worksheet_main = workbook.add_worksheet('main')

#prepare date
categories = []
print ("total count = %s" %(db.ebay_Female.count()))
for cat in ebay_constants.categories_keywords:
    items = db.ebay_Female.find({'categories': cat}).count()
    new_items = db.ebay_Female.find({'categories': cat, 'download_data.first_dl': today_date}).count()
    instock = db.ebay_Female.find({'categories': cat, 'status.instock': True}).count()
    out = db.ebay_Female.find({'categories': cat, 'status.instock': False}).count()
    categories.append([cat, items,new_items, instock, out])

# #create headers
bold = workbook.add_format({'bold': True, })

# worksheet_main.write(0, 0, 'CATEGORIES', bold)
# worksheet_main.write(0, 1,      'COUNT', bold)
#
# # Start from the first cell. Rows and columns are zero indexed.
# row = 1
# col = 0
worksheet_main.set_column('B:F',15)
worksheet_main.add_table('B2:F'+str(len(categories)+5),
                         {'data' : categories,
                          'total_row': True,
                          'columns': [{'header': 'Category', 'total_string': 'Total'},
                                      {'header': 'items',    'total_function' : 'sum'},
                                      {'header': 'new items','total_function' : 'sum'},
                                      {'header': 'instock',  'total_function' : 'sum'},
                                      {'header': 'out of stock', 'total_function' : 'sum'}],
                          'banded_columns': True,
                          'banded_rows': False,
                          'header_row': True})
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