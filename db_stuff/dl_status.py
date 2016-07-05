"""
run automatically every 15 minutes
check which downloads are running
display results on drive

need to add place holders to the download scripts
everyday open a item in collection dl_status
item = {"date": today,
        "ebay_male" : ebay_statue, - notice that all ebay downloads are running in parallel
        "ShopStyle_Female": ShopStyle_Female_status...
*later i might add extra info
download status can have the following stages:
    starting on ?
    working - in this case there will be an extra check - x percent complete - dl_version new / total count
    finishing up - archiving + drive
    done

    if failed there will be a FAILED error

note that Xlsxwriter can only create new files - it cannot read or modify existing files
 - thats why every run i am creating a new file
"""

import xlsxwriter
from ..Yonti import drive
from .. import constants
import pymongo
import argparse
import sys
from datetime import datetime,timedelta
now = datetime.now()
now_date = datetime.date(now)
current_date = str(now_date)
last2weeks = str(now_date-timedelta(days=15))
db = constants.db
dl_status=db.download_status

def createItem():
    existing = dl_status.find_one({"date":current_date})
    if existing:
        print("item already exists")
        return
    else:
        item = {"date": current_date,
                "collections":{
                "ebay_Female_US": {"status":"Starting on 15:00", "notes":"","EFT":"midnight"},
                "ebay_Male_US": {"status":"Starting on 15:00", "notes":"","EFT":"midnight"},
                "recruit_Female": {"status": "Starting on 20:00", "notes": "", "EFT": "midnight"},
                "recruit_Male": {"status": "Starting on 20:00", "notes": "", "EFT": "midnight"},
                "ShopStyle_Female": {"status":"Starting on 03:05", "notes":"","EFT":"7:00 AM"},
                "ShopStyle_Male": {"status":"Starting on 09:00", "notes":"","EFT":"9:00 AM"},
                "GangnamStyle_Female": {"status":"Starting on 06:00", "notes":"","EFT":"10:00 AM"},
                "GangnamStyle_Male": {"status":"Starting on 12:00", "notes":"","EFT":"12:00 AM"}}}

    dl_status.insert_one(item)
    print("new item inserted")
    return

def flatenDict(info):
    infoList = []
    keys = info["collections"].keys()
    keys.sort()
    for key in keys:
        status = info["collections"][key]["status"]
        if status == "Working":
            updated_count = db[key].find({'download_data.dl_version': current_date}).count()
            total = db[key].count()
            percent = int(100 * updated_count / total)
            notes = str(percent) + "% is already done"
        elif status == "Done":
            count = info['collections'][key]["notes"]
            notes = str(count) + " new items dl today"
        else:
            notes = info['collections'][key]["notes"]
        try:
            eft = info['collections'][key]["EFT"]
        except:
            eft = "TBD"
        item = [key, status, notes, eft]
        infoList.append(item)

    # for collection in ["ebay_","ShopStyle_","GangnamStyle_","recruit_"]:
    #     for gender in ["Male","Female","Unisex"]:#,"Tees"]:
    #         if gender in ["Unisex", "Tees"] and collection != "ebay_":
    #             continue
    #         key = collection + gender
    #         print (key)
    #
    return infoList


def addCollection2Collection_list(col_name, starting_time='12:34', ampm='am'):
    item = db.download_status.find_one({'date': current_date})
    collection_list = item['collections']
    if col_name in collection_list.keys():
        print ' already in keys'
        return
    starting_time_str = 'Starting on %s %s' %(starting_time,ampm)
    new_collection = {col_name:{'notes':'', 'status':starting_time_str }}
    collection_list.append(new_collection)
    db.download_status.update_one({'_id':item['_id']},{'$set':{'collections':collection_list}})
    print 'Done!'


def checkStatus():
    path2file = '/home/developer/yonti/dl_status.xlsx'
    workbook = xlsxwriter.Workbook(path2file)
    bold = workbook.add_format({'bold': True})

    lasts_days_info = dl_status.find({'date':{'$gte':last2weeks}}).sort('date',pymongo.DESCENDING)
    todays = workbook.add_worksheet('today')
    for daily_info in lasts_days_info:
        dl_date = daily_info['date']
        if  dl_date == current_date:
            current_worksheet = todays
            current_worksheet.write(11, 1, 'last check', bold)
            hour = str(now.hour)
            minute = now.minute
            if minute<10:
                minute = "0"+str(minute)
            else:
                minute=str(minute)
            current_worksheet.write(11, 2, " %s:%s" %(hour,minute), bold)
        else:
            current_worksheet= workbook.add_worksheet(dl_date)
        current_worksheet.write(0, 1, 'date', bold)
        current_worksheet.write(0, 2, dl_date, bold)
        dict2list = flatenDict(daily_info)
        keys = daily_info["collections"].keys()
        count = len(keys)
        current_worksheet.set_column('B:E', 30)
        tablesize = 'B3:E'+str(count+3)
        current_worksheet.add_table(tablesize,
                                    {'data': dict2list,
                                     'columns': [{'header': 'Collection Name'},
                                                 {'header': 'Download Status'},
                                                 {'header': 'Notes'},
                                                 {'header': 'Estimated Finishing Time '}],
                                     'banded_columns': True,
                                     'banded_rows': True})


    workbook.close()

    print ('uploading to drive...')
    files = ('Download_Status', path2file, True)
    res = drive.upload2drive(files)

    if res:
        print('file uploaded!')
    else:
        print ('error while uploading!')


def getUserInput():
    parser = argparse.ArgumentParser(description='"@@@ Downloads Status @@@')
    parser.add_argument('-m', '--mode',default="check", dest= "mode",
                        help='choose (regular) check or create (new item - delete items older than one month)')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    user_input = getUserInput()
    mode = user_input.mode
    if mode == 'create':
        createItem()
    else:
        checkStatus()
        print ("Status Checker Finished!!!")
    sys.exit(0)