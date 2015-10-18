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
    txt2 = 'date: ' + str(stats['date']) + \
           '\nitems_downloaded: ' + str(stats['items_downloaded']) + \
           '\nexisting_items: ' + str(stats['existing_items']) + \
           '\nnew_items:' + str(stats['new_items']) + \
           '\nitems_from_archive:' + str(stats['returned_from_archive']) + \
           '\nitems_sent_to_archive:' + str(stats['sent_to_archive']) + \
           '\ndl_duration(hours):' + str(stats['total_dl_time(hours)']) + \
           '\n\nitems_by_category:\n'
    for i in constants.db_relevant_items:
        txt2 = txt2 + i + ': ' + str(stats[i]) + '\n'
    #        "Copy %s\n\n" \
    #        "Go to %s\n\n" \
    #        "Pick the top 4 and save.\n\n" \
    #        "Thanks & Good luck!" % (image_url, trendi_url)
    # txt =
    msg = MIMEText(txt1 + txt2, 'plain')

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    # server.set_debuglevel(True)  # show communication with the server
    try:
        server.login('yonti0@gmail.com', "Hub,hKuhiPryh")
        server.sendmail(sender, yonti, msg.as_string())
        print "sent"
    except:
        print "error"
    finally:
        server.quit()


# trendi_url = 'http://extremeli.trendi.guru/demo/TrendiMatchEditor/matcheditor.html'
# image_url = 'http://image.gala.de/v1/cms/Mr/style-mandy-capristo-okt14-ge_7901219-ORIGINAL-imageGallery_standard.jpg?v=10333950'
# send_image_mail(trendi_url, image_url)

def download_stats():
    dl_data = db.download_data.find()[0]
    date = dl_data['current_dl']
    stats = {'date': date,
             'items_downloaded': dl_data['items_downloaded'],
             'existing_items': dl_data['existing_items'],
             'new_items': dl_data['new_items'],
             'items_from_archive': dl_data['returned_from_archive'],
             'items_sent_to_archive': dl_data['sent_to_archive'],
             'dl_duration(hours)': dl_data['total_dl_time(hours)'],
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
    # update_db.run_by_category(type="DAILY")
    download_stats()
    # e_mail
