__author__ = 'Nadav Paz'
'''
# # Import smtplib for the actual sending function
# import smtplib
#
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart
#
#
# def send_image_mail(trendi_url, image_url):
#
#     # me = 'nadav@trendiguru.com'
#     # lior = 'lior@trendiguru.com'
#     # kyle = 'kyle@trendiguru.com'
#     # jeremy = 'jeremy@trendiguru.com'
#     sender= 'yonti0@gmail.com'
#     yonti = 'yontilevin@gmail.com'
#     # recipient = 'members@trendiguru.com'
#     # Open a plain text file for reading.  For this example, assume that
#     msg = MIMEMultipart('alternative')
#     msg['Subject'] = 'A new image was uploaded!'
#     msg['From'] = sender
#     msg['To'] = yonti #recipient
#     text = "Hello TG member!\n\n" \
#            "There is a new image waiting to you.\n\n" \
#            "Copy %s\n\n" \
#            "Go to %s\n\n" \
#            "Pick the top 4 and save.\n\n" \
#            "Thanks & Good luck!" % (image_url, trendi_url)
#
#     part1 = MIMEText(txt1 + txt2, 'plain')
#     # html = """\
#     # <html>
#     #   <head></head>
#     #   <body>
#     #     <p>Hi!<br>
#     #        How are you?<br>
#     #        Here is the <a href="https://www.python.org">link</a> you wanted.
#     #     </p>
#     #   </body>
#     # </html>
#     # """
#     # part2 = MIMEText(html, 'html')
#     msg.attach(part1)
#     msg.attach(part2)
#     server = smtplib.SMTP('smtp.gmail.com', 587)
#     server.starttls()
#     # server.set_debuglevel(True)  # show communication with the server
#     try:
#         server.login('yonti0@gmail.com', "Hub,hKuhiPryh")
#         server.sendmail(sender, [yonti], msg.as_string())
#         print "sent"
#     except:
#         print "error"
#     finally:
#         server.quit()
#
#     # part1 = MIMEText(text, 'plain')
#     # msg.attach(part1)
#     #
#     # s = smtplib.SMTP('localhost')
#     # s.sendmail(sender, [me, yonti], msg.as_string())
#     # s.quit()
#
#
# trendi_url = 'http://extremeli.trendi.guru/demo/TrendiMatchEditor/matcheditor.html'
# image_url = 'http://image.gala.de/v1/cms/Mr/style-mandy-capristo-okt14-ge_7901219-ORIGINAL-imageGallery_standard.jpg?v=10333950'
# send_image_mail(trendi_url, image_url)
'''
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


if __name__ == "__main__":
    # test the class here
    import smtplib

    sender = "yonti0@gmail.com"
    msg = MIMEMultipart('alternative')
    msg["Subject"] = "email test"
    msg["From"] = "yonti <" + sender + ">"
    msg["To"] = "yontilevin@gmail.com"
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    # server.set_debuglevel(True)  # show communication with the server
    # Create the body of the message (a plain-text and an HTML version).
    text = "Hi!\nHow are you?\nHere is the link you wanted:\nhttp://www.python.org"
    html = """\
    <!DOCTYPE html>
    <html>
    <body>

    <table border="1" style="width:100%">
      <tr>
        <td>Jill</td>
        <td>Smith</td>
        <td>50</td>
      </tr>
      <tr>
        <td>Eve</td>
        <td>Jackson</td>
        <td>94</td>
      </tr>
      <tr>
        <td>John</td>
        <td>Doe</td>
        <td>80</td>
      </tr>
    </table>

    </body>
    </html>
    """

    # Record the MIME types of both parts - text/plain and text/html.
    part1 = MIMEText(text, 'plain')
    part2 = MIMEText(html, 'html')

    # Attach parts into message container.
    # According to RFC 2046, the last part of a multipart message, in this case
    # the HTML message, is best and preferred.
    msg.attach(part1)
    msg.attach(part2)

    try:
        server.login('yonti0@gmail.com', "Hub,hKuhiPryh")
        server.sendmail(sender, msg["To"], msg.as_string())
        print "mail sent"
    except:
        print "error"
    finally:
        server.quit()
