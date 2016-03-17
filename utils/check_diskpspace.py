__author__ = 'jeremy'

import subprocess
import smtplib
import sys
sender = 'the_guru@trendi.guru'
receivers = ['jeremy.rutman@gmail.com']

message = """From: the guru <the_guru@trendi.guru>
To: To Person <to@todomain.com>
Subject: disk overflow
"""


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
    if int(percentage)>10:
        print('high percent on {}!'.format(device))
        message = message + 'high percent use ({})on {}!'.format(percentage,device)
        try:
           smtpObj = smtplib.SMTP('localhost')
           smtpObj.sendmail(sender, receivers, message)
           print "Successfully sent email"
        except :
           print "Error: unable to send email: "+ str(sys.exc_info()[0])

#print device