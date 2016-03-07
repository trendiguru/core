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
for cat in ebay_constants.categories_keywords:
    count = db.ebay_Female.find({'categories':cat}).count()
    categories.append([cat, count])

# Start from the first cell. Rows and columns are zero indexed.
row = 0
col = 0

# Iterate over the data and write it out row by row.
for item, cost in (categories):
    worksheet_main.write(row, col,     item)
    worksheet_main.write(row, col + 1, cost)
    row += 1

# Write a total using a formula.
lastrow = row-1
worksheet_main.write(row, 0, 'Total')
worksheet_main.write(row, 1, '=SUM(B1:B'+str(lastrow)+')')

workbook.close()

print ('uploading to drive...')
files = (filename,path2file, True)
res = drive.upload2drive(files)
if res:
    print('files uploaded!')
else:
    print ('error while uploading!')