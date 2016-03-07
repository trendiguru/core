import xlsxwriter
from .Yonti import drive

filename = 'ebay'
path2file = '/home/developer/yonti/ebay.xlsx'
workbook = xlsxwriter.Workbook(path2file)
worksheet_main = workbook.add_worksheet('main')

worksheet_main.write('A1', 'Hello world')

workbook.close()

print ('uploading to drive...')
files = [(filename,path2file, True)]
res = drive.upload2drive(files)
if res:
    print('files uploaded!')
else:
    print ('error while uploading!')