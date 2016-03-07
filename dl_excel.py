import xlsxwriter
from .Yonti import drive
from . import constants
from . import ebay_constants
db = constants.db

filename = 'ebay'
path2file = '/home/developer/yonti/ebay.xlsx'
workbook = xlsxwriter.Workbook(path2file)
worksheet_main = workbook.add_worksheet('main')

#prepare date
categories = []
print ("total count = %s" %(db.ebay_Female.count()))
for cat in ebay_constants.categories_keywords:
    count = db.ebay_Female.find({'categories':cat}).count()
    categories.append([cat, count])

# #create headers
# bold = workbook.add_format({'bold': True})
# worksheet_main.write(0, 0, 'CATEGORIES', bold)
# worksheet_main.write(0, 1,      'COUNT', bold)
#
# # Start from the first cell. Rows and columns are zero indexed.
# row = 1
# col = 0

worksheet_main.add_table('B2:C'+str(len(categories)), {'data' : categories,
                                                       'total_row': True,
                                                       'columns':[{'header': 'categories'},
                                                                  {'header': 'count'}
                                                                  ],
                                                       'first_column': True,
                                                       'last_column' : True})
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
    print('files uploaded!')
else:
    print ('error while uploading!')