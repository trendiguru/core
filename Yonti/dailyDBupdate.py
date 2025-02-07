__author__ = 'yonatan'

"""
add description
"""

import smtplib

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import getShopStyleDB
import constants

db = constants.db


def email(stats, coll):
    # me = 'nadav@trendiguru.com'
    # lior = 'lior@trendiguru.com'
    # kyle = 'kyle@trendiguru.com'
    # jeremy = 'jeremy@trendiguru.com'
    yonti = 'yontilevin@gmail.com'
    sender = 'yontiforall@gmail.com'
    #
    recipient = 'yontilevin@gmail.com'  # 'team@trendiguru.com'

    # Open a plain text file for reading.  For this example, assume that
    msg = MIMEMultipart('alternative')
    msg['Subject'] = 'Daily DB download&update! - ' + coll
    msg['From'] = sender
    msg['To'] = yonti

    txt2 = '<h1><mark>' + coll + '</mark></h1><br>' \
                                 '<h3> date:\t' + str(stats['date']) + '</h3>\n<h3>' + \
           'items downloaded:\t' + str(stats['items_downloaded']) + '</h3>\n<h3>' + \
           'new items:\t' + str(stats['new_items']) + '</h3>\n<h3>' + \
           'insert errors:\t' + str(stats['errors']) + '</h3>\n<h3>' + \
           'dl duration(min):\t' + str(stats['dl_duration(min)'])[:5] + '</h3>\n<h3>' + \
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
    <style>
    table, th, td {
        border: 1px solid black;
        border-collapse: collapse;
    }
    th, td {
        padding: 5px;
    }
    </style>
</head>
<body>
<div class="container">"""
    html = html + txt2 + """
    <table  style="width:40%">
    <thead>
      <tr>
        <th>Category</th>
        <th>total items</th>
        <th>new items</th>
      </tr>
    </thead>
    <tbody>
    """
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
        server.login('yontiforall@gmail.com', "Hub,hKuhiZck")
        server.sendmail(sender, recipient, msg.as_string())  # [recipient, yonti], msg.as_string())
        print "sent"
    except:
        print "error"
    finally:
        server.quit()


def stats_and_mail(collection):
    dl_data = db.download_data.find({"criteria": collection})[0]
    date = dl_data['current_dl']
    stats = {'date': date,
             'items_downloaded': dl_data['items_downloaded'],
             'new_items': dl_data['new_items'],
             'dl_duration(min)': dl_data['total_dl_time(min)'],
             'errors': dl_data['errors'],
             'items_by_category': {}}
    # for i in constants.db_relevant_items:
    #     if i == 'women' or i == 'women-cloth':
    #         continue
    #     stats['items_by_category'][i] = {'total': db[collection].find({'categories.id': i}).count(),
    #                                      'new': db[collection].find({'$and': [{'categories.id': i},
    #                                                                        {'download_data.first_dl': date}]}).count()}
    for i in constants.paperdoll_relevant_categories:
        stats['items_by_category'][i] = {'total': db[collection].find({'categories': i}).count(),
                                         'new': db[collection].find({'$and': [{'categories': i},
                                                                           {'download_data.first_dl': date}]}).count()}
    email(stats, collection)
    # with open(date + '.txt', 'w') as outfile:
    #     json.dump(stats, outfile)


if __name__ == "__main__":
    # if len(sys.argv) == 1:
    #     collection = "products"
    # else:
    #     collection = sys.argv[1]
    # print "\n@@@ The Daily DB Updater @@@\n you choose to update the " + collection + " collection"
    # while True:
    #     try:
    #         x = raw_input("Download or Statistics? (D/S)")
    #         break
    #     except ValueError:
    #         print "Oops! wrong input."
    #         y = raw_input("Try again?(Y/N)")
    #         if y is "Y" or y is "y":
    #             continue
    #         else:
    #             exit()
    x = "d"
    collection = "products"
    if x is "D" or x is "d":
        update_db = getShopStyleDB.ShopStyleDownloader()
        update_db.db_download(collection)
    stats_and_mail(collection)

    print "Daily Update Finished!!!"
