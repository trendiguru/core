__author__ = 'Nadav Paz'

# Import smtplib for the actual sending function
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


def send_image_mail(url):

    me = 'nadav@trendiguru.com'
    lior = 'lior@trendiguru.com'
    kyle = 'kyle@trendiguru.com'
    jeremy = 'jeremy@trendiguru.com'

    # Open a plain text file for reading.  For this example, assume that
    msg = MIMEMultipart('alternative')
    msg['Subject'] = 'A new image was uploaded!'
    msg['From'] = me
    msg['To'] = me
    text = "Hello TG member!\n\n" \
           "There is a new image waiting to you.\n\n" \
           "Here is the link you wanted: %s\n\n" \
           "Thanks & Good luck!" % url
    part1 = MIMEText(text, 'plain')
    msg.attach(part1)

    s = smtplib.SMTP('localhost')
    s.sendmail(me, [me], msg.as_string())
    s.quit()


URL = 'http://extremeli.trendi.guru/demo/TrendiMatchEditor/matcheditor.html'
send_image_mail(URL)