from ftplib import FTP

us_params = {"url": "partnersw.ftp.ebaycommercenetwork.com",
             "user": 'p1129643',
             'password': '6F2lqCf4'}

# username, passwd are for the US - for other countries ftp codes check our google drive
def ftp_connection(params):
    ftp = FTP(params["url"])
    ftp.login(user=params["user"], passwd=params["password"])
    return ftp