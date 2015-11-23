__author__ = 'yonatan'

import csv
import StringIO
import zipfile
import time

import requests

from constants import db

headers = {"FK-Affiliate-Id": "kyletrend", "FK-Affiliate-Token": "74deca1b038141e2996cd3f170445fbb"}
url = "https://affiliate-api.flipkart.net/affiliate/api/kyletrend.json"

r1 = requests.get(url=url, headers=headers)

r1 = r1.json()

url2 = r1["apiGroups"]["affiliate"]["rawDownloadListings"]["womens_clothing"]["availableVariants"]["v0.1.0"]["get"]

r2 = requests.get(url=url2, headers=headers)
r2zip = zipfile.ZipFile(StringIO.StringIO(r2.content))
r2zip.extractall()
csv_file = open(r2zip.infolist()[0].filename, 'rb')
time.sleep(120)
DB = csv.reader(csv_file)
time.sleep(120)
for row in DB:
    if row['categories'] == "Apparels>Women>Ethnic Wear>Ethnic Bottoms>Harem Pants - women pants"
        db.flipkart.insert_one(row)

DB = csv.reader(DB_csv)
list = []
print DB[0]
