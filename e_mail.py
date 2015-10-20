__author__ = 'yonatan'

"""
1. run getShopStyleDB.py
2. do stats
3. mail the output to relevent
"""

import json
import smtplib

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import constants

db = constants.db


def email(stats):
    # me = 'nadav@trendiguru.com'
    # lior = 'lior@trendiguru.com'
    # kyle = 'kyle@trendiguru.com'
    # jeremy = 'jeremy@trendiguru.com'
    yonti = 'yontilevin@gmail.com'
    sender = 'Notifier@trendiguru.com'
    # recipient = 'members@trendiguru.com'

    # Open a plain text file for reading.  For this example, assume that
    msg = MIMEMultipart('alternative')
    msg['Subject'] = 'Daily DB download&update!'
    msg['From'] = sender
    msg['To'] = yonti
    txt1 = "Hello TG member!\n\n" \
           "This is your daily DB update - so get ready to be amazed...\n\n"
    txt2 = '<h3>date:\t' + str(stats['date']) + \
           '\nitems downloaded:\t' + str(stats['items_downloaded']) + \
           '\nexisting items:\t' + str(stats['existing_items']) + \
           '\nnew items:\t' + str(stats['new_items']) + \
           '\nitems from archive:\t' + str(stats['items_from_archive']) + \
           '\nitems sent to archive:\t' + str(stats['items_sent_to_archive']) + '\n\nitems by category:<h3>'
    # '\ndl duration(hours):\t' + str(stats['dl_duration(hours)'])[:5] + \

    categories = ""
    for i in constants.db_relevant_items:
        if i == 'women' or i == 'women-clothes':
            continue
        total = str(stats['items_by_category'][i]["total"])
        new = str(stats['items_by_category'][i]["new"])
        line = "<tr>\n\t<th>" + i + "</th>\n\t<th>" + total + "</th>\n\t<th>" + new + "</th>\n</tr>\n"
        categories = categories + line

    html = """\
    <html>
    <head>
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
    <body>"""
    html + txt2 + """
    <table style="width:100%">
    """
    html = html + categories + """
    </table>

    </body>
    </html>
    """
    part1 = MIMEText(txt1 + html, 'html')
    msg.attach(part1)
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    # server.set_debuglevel(True)  # show communication with the server
    try:
        server.login('yonti0@gmail.com', "Hub,hKuhiPryh")
        server.sendmail(sender, [yonti], msg.as_string())
        print "sent"
    except:
        print "error"
    finally:
        server.quit()


def download_stats():
    dl_data = db.download_data.find()[0]
    date = dl_data['current_dl']
    stats = {'date': date,
             'items_downloaded': dl_data['items_downloaded'],
             'existing_items': dl_data['existing_items'],
             'new_items': dl_data['new_items'],
             'items_from_archive': dl_data['returned_from_archive'],
             'items_sent_to_archive': dl_data['sent_to_archive'],
             # 'dl_duration(hours)': dl_data['total_dl_time(hours)'],
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
    # update_db = getShopStyleDB.ShopStyleDownloader()
    # update_db.run_by_category(type="FULL")
    # time.sleep(14440)
    download_stats()
    print "Daily Download Finished!!!"
