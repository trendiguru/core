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
import urllib2
import urlparse
import re

from email.MIMEText import MIMEText
from email.MIMEImage import MIMEImage
from email.MIMEMultipart import MIMEMultipart


class HtmlMail:
    def __init__(self, location, encoding="iso-8859-1"):
        self.location = location
        if location.find("http://") == 0:
            self.is_http = True
        else:
            self.is_http = False

        self.encoding = encoding

        self.p1 = re.compile("(<img.*?src=\")(.*?)(\".*?>)", re.IGNORECASE | re.DOTALL)
        self.p2 = re.compile("(<.*?background=\")(.*?)(\".*?>)", re.IGNORECASE | re.DOTALL)
        self.p3 = re.compile("(<input.*?src=\")(.*?)(\".*?>)", re.IGNORECASE | re.DOTALL)

        self.img_c = 0

    def set_log(self, log):
        self.log = log

    def _handle_image(self, matchobj):
        img = matchobj.group(2)
        if not self.images.has_key(img):
            self.img_c += 1
            self.images[img] = "dazoot-img%d" % self.img_c
        return "%scid:%s%s" % (matchobj.group(1), self.images[img], matchobj.group(3))

    def _parse_images(self):
        self.images = {}
        self.content = self.p1.sub(self._handle_image, self.content)
        self.content = self.p2.sub(self._handle_image, self.content)
        self.content = self.p3.sub(self._handle_image, self.content)
        return self.images

    def _read_image(self, imglocation):
        if self.is_http:
            img_url = urlparse.urljoin(self.location, imglocation)
            content = urllib2.urlopen(img_url).read()
            return content
        else:
            return file(imglocation, "rb").read()

    def get_msg(self):
        if self.is_http:
            content = urllib2.urlopen(self.location).read()
        else:
            content = file(self.location, "r").read()
        self.content = content

        msg = MIMEMultipart("related")
        images = self._parse_images()

        tmsg = MIMEText(self.content, "html", self.encoding)
        msg.attach(tmsg)

        for img in images.keys():
            img_content = self._read_image(img)
            img_msg = MIMEImage(img_content)
            img_type, img_ext = img_msg["Content-Type"].split("/")

            del img_msg["MIME-Version"]
            del img_msg["Content-Type"]
            del img_msg["Content-Transfer-Encoding"]

            img_msg.add_header("Content-Type", "%s/%s; name=\"%s.%s\"" % (img_type, img_ext, images[img], img_ext))
            img_msg.add_header("Content-Transfer-Encoding", "base64")
            img_msg.add_header("Content-ID", "<%s>" % images[img])
            img_msg.add_header("Content-Disposition", "inline; filename=\"%s.%s\"" % (images[img], img_ext))
            msg.attach(img_msg)

        return msg


if __name__ == "__main__":
    # test the class here
    import smtplib

    hm = HtmlMail("http://buymelaughs.com/wp-content/uploads/2014/01/Funny-Babies-Pictures-2.jpg")
    sender = "yonti0@gmail.com"
    msg = hm.get_msg()
    msg["Subject"] = "email test"
    msg["From"] = "yonti <" + sender + ">"
    msg["To"] = "yontilevin@gmail.com"
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    # server.set_debuglevel(True)  # show communication with the server
    try:
        server.login('yonti0@gmail.com', "Hub,hKuhiPryh")
        server.sendmail(sender, msg["To"], msg.as_string())
        print "mail sent"
    except:
        print "error"
    finally:
        server.quit()
