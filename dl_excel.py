import xlsxwriter
from .Yonti import drive
from . import constants
db = constants.db

filename = 'ebay'
path2file = '/home/developer/yonti/ebay.xlsx'
workbook = xlsxwriter.Workbook(path2file)
worksheet_main = workbook.add_worksheet('main')

#prepare date
expenses = (
    ['Rent', 1000],
    ['Gas',   100],
    ['Food',  300],
    ['Gym',    50],
)

# Start from the first cell. Rows and columns are zero indexed.
row = 0
col = 0

# Iterate over the data and write it out row by row.
for item, cost in (expenses):
    worksheet_main.write(row, col,     item)
    worksheet_main.write(row, col + 1, cost)
    row += 1

# Write a total using a formula.
worksheet_main.write(row, 0, 'Total')
worksheet_main.write(row, 1, '=SUM(B1:B4)')

workbook.close()

print ('uploading to drive...')
files = (filename,path2file, True)
res = drive.upload2drive(files)
if res:
    print('files uploaded!')
else:
    print ('error while uploading!')