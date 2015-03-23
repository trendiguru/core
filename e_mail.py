__author__ = 'Nadav Paz'

# Import smtplib for the actual sending function
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


def send_image_mail(trendi_url, image_url):

    me = 'nadav@trendiguru.com'
    lior = 'lior@trendiguru.com'
    kyle = 'kyle@trendiguru.com'
    jeremy = 'jeremy@trendiguru.com'
    sender = me
    recipient = lior
    # Open a plain text file for reading.  For this example, assume that
    msg = MIMEMultipart('alternative')
    msg['Subject'] = 'A new image was uploaded!'
    msg['From'] = sender
    msg['To'] = recipient
    text = "Hello TG member!\n\n" \
           "There is a new image waiting to you.\n\n" \
           "Copy %s\n\n" \
           "Go to %s\n\n" \
           "Pick the top 4 and save.\n\n" \
           "Thanks & Good luck!" % (image_url, trendi_url)
    part1 = MIMEText(text, 'plain')
    msg.attach(part1)

    s = smtplib.SMTP('localhost')
    s.sendmail(sender, [recipient], msg.as_string())
    s.quit()


trendi_url = 'http://extremeli.trendi.guru/demo/TrendiMatchEditor/matcheditor.html'
image_url = 'http://image.gala.de/v1/cms/Mr/style-mandy-capristo-okt14-ge_7901219-ORIGINAL-imageGallery_standard.jpg?v=10333950'
send_image_mail(trendi_url, image_url)