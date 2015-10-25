__author__ = 'yonatan'

"""
add description
"""

import json
import smtplib
import time

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import getShopStyleDB
import constants

db = constants.db


def email(stats):
    # me = 'nadav@trendiguru.com'
    # lior = 'lior@trendiguru.com'
    # kyle = 'kyle@trendiguru.com'
    # jeremy = 'jeremy@trendiguru.com'
    yonti = 'yontilevin@gmail.com'
    sender = 'yonti0@gmail.com'
    #
    recipient = 'team@trendiguru.com'

    # Open a plain text file for reading.  For this example, assume that
    msg = MIMEMultipart('alternative')
    msg['Subject'] = 'Daily DB download&update!'
    msg['From'] = sender
    msg['To'] = yonti

    txt2 = '<h3> date:\t' + str(stats['date']) + '</h3>\n<h3>' + \
           'items downloaded:\t' + str(stats['items_downloaded']) + '</h3>\n<h3>' + \
           'existing items:\t' + str(stats['existing_items']) + '</h3>\n<h3>' + \
           'new items:\t' + str(stats['new_items']) + '</h3>\n<h3>' + \
           'items from archive:\t' + str(stats['items_from_archive']) + '</h3>\n<h3>' + \
           'items sent to archive:\t' + str(stats['items_sent_to_archive']) + '</h3>\n<h3>' + \
           'insert errors:\t' + str(stats['errors']) + '</h3>\n<h3>' + \
           'dl duration(hours):\t' + str(stats['dl_duration(hours)'])[:5] + '</h3>\n<h3>' + \
           '</h3>\n<h3>' + '</h3>\n<h3>' + 'items by category:</h3>\n' + '</h3>\n<h3>'

    categories = ""
    for i in constants.db_relevant_items:
        if i == 'women' or i == 'women-clothes':
            continue
        total = str(stats['items_by_category'][i]["total"])
        new = str(stats['items_by_category'][i]["new"])
        line = "<tr>\n\t<th>" + i + "</th>\n\t<th>" + total + "</th>\n\t<th>" + new + "</th>\n</tr>\n"
        categories = categories + line

    # html = """\
    # <html>
    # <head>
    # <style>
    # table, th, td {
    #     border: 1px solid black;
    #     border-collapse: collapse;
    # }
    # th, td {
    #     padding: 5px;
    # }
    # </style>
    # </head>
    # <body>"""
    #
    html = """
    <!DOCTYPE html>
<html lang="en">
<head>
  <title>Bootstrap Example</title>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="stylesheet" href="http://maxcdn.bootstrapcdn.com/bootstrap/3.3.5/css/bootstrap.min.css">
  <script src="https://ajax.googleapis.com/ajax/libs/jquery/1.11.3/jquery.min.js"></script>
  <script src="http://maxcdn.bootstrapcdn.com/bootstrap/3.3.5/js/bootstrap.min.js"></script>
</head>
<body>
<div class="container">"""
    html = html + txt2 + """
    <table class="table table-bordered" style="width:40%">
    <thead>
      <tr>
        <th>Category</th>
        <th>total items</th>
        <th>new items</th>
      </tr>
    </thead>
    <tbody> """
    html = html + categories + """
    </tbody>
    </table>
    </div>
    </body>
    </html>
    """
    part1 = MIMEText(html, 'html')
    msg.attach(part1)
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    # server.set_debuglevel(True)  # show communication with the server
    try:
        server.login('yonti0@gmail.com', "Hub,hKuhiPryh")
        server.sendmail(sender, [yonti], msg.as_string())  # [recipient, yonti], msg.as_string())
        print "sent"
    except:
        print "error"
    finally:
        server.quit()


def wait_for(dl_data):
    x = raw_input("waitfor enabled? (Y/N)")
    if x == "n" or x == "N":
        return
    total_items = db.products.find().count()
    downloaded_items = dl_data["items_downloaded"]
    new_items = dl_data["new_items"]
    insert_errors = dl_data["errors"]
    sub = downloaded_items - insert_errors
    if total_items > sub:
        time.sleep(new_items / 100)
    else:
        check = 0
        while sub > total_items:
            if check > 60:
                break
            print "\ncheck number " + str(check)
            print "\nfp workers didn't finish yet\nWaiting 10 min before checking again\n"
            check += 1
            print "check number" + str(check)
            time.sleep(600)
            total_items = db.products.find().count()
            insert_errors = dl_data["errors"]
            sub = downloaded_items - insert_errors


def stats_and_mail():
    dl_data = db.download_data.find()[0]
    date = dl_data['current_dl']
    wait_for(dl_data)
    stats = {'date': date,
             'items_downloaded': dl_data['items_downloaded'],
             'existing_items': dl_data['existing_items'],
             'new_items': dl_data['new_items'],
             'items_from_archive': dl_data['returned_from_archive'],
             'items_sent_to_archive': dl_data['sent_to_archive'],
             'dl_duration(hours)': dl_data['total_dl_time(hours)'],
             'errors': dl_data['errors'],
             'items_by_category': {}}
    for i in constants.db_relevant_items:
        if i == 'women' or i == 'women-cloth':
            continue
        stats['items_by_category'][i] = {'total': db.products.find({'categories.id': i}).count(),
                                         'new': db.products.find({'$and': [{'categories.id': i},
                                                                           {'download_data.first_dl': date}]}).count()}
    email(stats)
    with open(date + '.txt', 'w') as outfile:
        json.dump(stats, outfile)


if __name__ == "__main__":
    print "\n@@@ The Daily DB Updater @@@"
    while True:
        try:
            x = raw_input("Download or Statistics? (D/S)")
            break
        except ValueError:
            print "Oops! wrong input."
            y = raw_input("Try again?(Y/N)")
            if y is "Y" or y is "y":
                continue
            else:
                exit()

    if x is "D" or x is "d":
        update_db = getShopStyleDB.ShopStyleDownloader()
        update_db.run_by_category()
    stats_and_mail()

    print "Daily Update Finished!!!"
