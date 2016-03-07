import xlsxwriter
from .Yonti import drive

filename = '/home/developer/yonti/ebay.xlsx'
workbook = xlsxwriter.Workbook(filename)
worksheet_main = workbook.add_worksheet('main')

worksheet_main.write('A1', 'Hello world')

workbook.close()

print ('uploading to drive...')
files = [(filename, True)]
res = drive.upload2drive(files)
if res:
    print('files uploaded!')
else:
    print ('error while uploading!')