__author__ = 'jeremy'
#stolen from nadav

import subprocess
import smtplib
import sys

import time
import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import datetime
import subprocess


nadav = 'nadav@trendiguru.com'
lior = 'lior@trendiguru.com'
kyle = 'kyle@trendiguru.com'
jeremy = 'jeremy@trendiguru.com'
yonti = 'yontilevin@gmail.com'
sender = 'Notifier@trendiguru.com'
all = 'members@trendiguru.com'




sender = 'the_guru@trendi.guru'
receivers = [nadav,lior,jeremy,yonti]
#receivers = 'jeremy.rutman@gmail.com'

message = """From: the guru <the_guru@trendi.guru>
To: To Person <to@todomain.com>
Subject: disk overflow
"""




def email(message_text, title, recipients):
    msg = MIMEMultipart('alternative')
    msg['Subject'] = title
    msg['From'] = sender
    msg['To'] = ", ".join(recipients)

    txt = '<h3> disk getting near full </h3>\n'
    txt = txt + '<h3> '+str(message_text)+' </h3>\n'

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
    html += txt

    part1 = MIMEText(html, 'html')
    msg.attach(part1)
    server = smtplib.SMTP('smtp-relay.gmail.com', 587)
    server.starttls()
    server.sendmail(sender, recipients, msg.as_string())
    server.quit()



df = subprocess.Popen(["df"], stdout=subprocess.PIPE)
output = df.communicate()[0]
lines = output.split("\n")
relevant_lines = lines[1:-1] #first line is heading, last line is /n or something
for line in relevant_lines:
    #print line
    #elements = line.split()
    device, size, used, available, percent, mountpoint = line.split()
    print('device {} size {} u {} a{} p {} m {}'.format(device, size, used, available, percent, mountpoint))
    percentage = percent.split('%')[0]
   # print percentage
    if int(percentage)>90:
        print('high percent on {}!'.format(device))
        host = str(socket.gethostname())
        message = 'high percent disk use ({})% on device {} of host {}!'.format(percentage,device,host)

        try:
            email(message, 'high disk use on '+str(device)+' host:'+host, receivers)

            print "Successfully sent email to "+str(receivers)
        except :
            print "Error: unable to send email: "+ str(sys.exc_info()[0])

#print device