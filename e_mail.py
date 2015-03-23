__author__ = 'Nadav Paz'

# Import smtplib for the actual sending function
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Import the email modules we'll need


# Open a plain text file for reading.  For this example, assume that
# the text file contains only ASCII characters.
"""fp = open('textfile.txt', 'rb')
# Create a text/plain message
msg = MIMEText(fp.read())
fp.close()"""

# me == the sender's email address
# you == the recipient's email address
me = 'nadav@trendiguru.com'
lior = 'lior@trendiguru.com'
kyle = 'kyle@trendiguru.com'
jeremy = 'jeremy@trendiguru.com'
sub = 'Yay! a new image is waiting for you!'
msg = MIMEMultipart('alternative')
msg['Subject'] = 'A new image was uploaded!'  # %s % textfile
msg['From'] = me
text = "Hello TG member!\n" \
       "There is a new image waiting to you.\n" \
       "Here is the link you wanted: %s\n " \
       "Thanks & Good luck!"
part1 = MIMEText(text, 'plain')
msg.attach(part1)
# Send the message via our own SMTP server, but don't include the
# envelope header.
s = smtplib.SMTP('localhost')
s.sendmail(me, [me], msg.as_string())
s.quit()