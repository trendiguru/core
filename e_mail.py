__author__ = 'Nadav Paz'

# Import smtplib for the actual sending function
import smtplib

# Import the email modules we'll need
from email.mime.text import MIMEText
from email.parser import Parser
"""headers = Parser().parsestr('From: <nadav@trendiguru.com>\n'
                            'To: <someone_else@example.com>\n'
                            'Subject: Test message\n'
                            '\n'
                            'Body would go here\n')
"""
# Open a plain text file for reading.  For this example, assume that
# the text file contains only ASCII characters.
fp = open('textfile.txt', 'rb')
# Create a text/plain message
msg = MIMEText(fp.read())
fp.close()

# me == the sender's email address
# you == the recipient's email address
me = 'nadav@trendiguru.com'
lior = 'lior@trendiguru.com'
kyle = 'kyle@trendiguru.com'
jeremy = 'jeremy@trendiguru.com'
sub = 'Yay! a new image is waiting for you!'
dict = {"Subject": [sub], "From": [me], "To": [me, lior, kyle, jeremy]}
msg['Subject'] = 'Subject'  # %s % textfile
msg['From'] = me

# Send the message via our own SMTP server, but don't include the
# envelope header.
s = smtplib.SMTP('localhost')
#for to in dict["To"]:
s.sendmail(me, [me, lior, jeremy, kyle], msg.as_string())
s.quit()