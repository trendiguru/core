__author__ = 'yonatan'

import smtplib

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import constants

db = constants.db


def email(stats, spec_coll=None):
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
    msg['Subject'] = 'Daily DB download&update! '
    msg['From'] = sender
    msg['To'] = yonti
    txt2 = ''
    for curr in stats:
        coll = spec_coll or curr["collection"]
        if curr['dl_duration(min)'] is str:
            duration = "still in process"
        else:
            duration = str(curr['dl_duration(min)'])

        txt2 = txt2 + '<h1><mark>' + coll + '</mark></h1>' \
                                            '<h3> date:\t' + str(curr['date']) + '</h3>\n<h3>' + \
               'items downloaded:\t' + str(curr['items_downloaded']) + '</h3>\n<h3>' + \
               'new items:\t' + str(curr['new_items']) + '</h3>\n<h3>' + \
               'insert errors:\t' + str(curr['errors']) + '</h3>\n<h3>' + \
               'dl duration(min):\t' + duration + '</h3>\n\n<h3>' + \
               'total_items:\t' + str(curr['total_items']) + '</h3>\n<h3>' + \
               'in stock:\t' + str(curr['instock']) + '</h3>\n<h3>' + \
               'out of stock:\t' + str(curr['out']) + '</h3><br>'

        # '</h3>\n<h3>' + '</h3>\n<h3>' + 'items by category:</h3>\n' + '</h3>\n<h3>'


    # categories = ""
    # for i in constants.db_relevant_items:
    #     if i == 'women' or i == 'women-clothes':
    #         continue
    #     total = str(stats['items_by_category'][i]["total"])
    #     new = str(stats['items_by_category'][i]["new"])
    #     line = "<tr>\n\t<th>" + i + "</th>\n\t<th>" + total + "</th>\n\t<th>" + new + "</th>\n</tr>\n"
    #     categories = categories + line

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
    <body>
        <div>"""
    html = html + txt2 + \
           """</div>
       </body>
       </html>
       """
    # """
    # <table  style="width:40%">
    # <thead>
    #   <tr>
    #     <th>Category</th>
    #     <th>total items</th>
    #     <th>new items</th>
    #   </tr>
    # </thead>
    # <tbody>
    # """
    # html = html + categories + """
    # </tbody>
    # </table>

    part1 = MIMEText(html, 'html')
    msg.attach(part1)
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    # server.set_debuglevel(True)  # show communication with the server
    try:
        server.login('yontiforall@gmail.com', "Hub,hKuhiZck")
        server.sendmail(sender, recipient, msg.as_string())  # [recipient, yonti], msg.as_string())
        print "email sent"
    except:
        print "email error"
    finally:
        server.quit()


def stats_and_mail():
    data = db.download_data.find()
    stats = []
    for x, dl_data in enumerate(data):
        date = dl_data['current_dl']
        stats.append({'collection': dl_data["criteria"],
                      'date': date,
                      'items_downloaded': dl_data['items_downloaded'],
                      'new_items': dl_data['new_items'],
                      'dl_duration(min)': dl_data['total_dl_time(min)'],
                      'errors': dl_data['errors'],
                      'total_items': dl_data["total_items"],
                      "instock": dl_data["instock"],
                      "out": dl_data["out"]})
        # 'items_by_category': {}}
    # for i in constants.db_relevant_items:
    #     if i == 'women' or i == 'women-cloth':
    #         continue
    #     stats['items_by_category'][i] = {'total': db[collection].find({'categories.id': i}).count(),
    #                                      'new': db[collection].find({'$and': [{'categories.id': i},
    #                                                                        {'download_data.first_dl': date}]}).count()}
    # for i in constants.paperdoll_relevant_categories:
    #     stats['items_by_category'][i] = {'total': db[collection].find({'categories': i}).count(),
    #                                      'new': db[collection].find({'$and': [{'categories': i},
    #                                                                        {'download_data.first_dl': date}]}).count()}

    email(stats)
    # with open(date + '.txt', 'w') as outfile:
    #     json.dump(stats, outfile)


if __name__ == "__main__":
    stats_and_mail()
