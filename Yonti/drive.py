from __future__ import print_function
from apiclient.discovery import build
from apiclient import errors
from httplib2 import Http
from oauth2client import file, client, tools
try:
    import argparse
    flags = argparse.ArgumentParser(parents=[tools.argparser]).parse_args()
except ImportError:
    flags = None

def upload2drive(FILES):
#FILES = [(filename, path2file, True/False),...]
    try:
        SCOPES = 'https://www.googleapis.com/auth/drive'
        store = file.Storage('/home/developer/python-packages/trendi/Yonti/storage.json')
        creds = store.get()
        if not creds or creds.invalid:
            flow = client.flow_from_clientsecrets('/home/developer/python-packages/trendi/Yonti/client_secret.json',
                                                  SCOPES)
            creds = tools.run_flow(flow, store, flags) \
                    if flags else tools.run(flow, store)
        DRIVE = build('drive', 'v2', http=creds.authorize(Http()))

        for filename,path2file,convert in FILES:
            metadata = {'title': filename,
                        'parents':[{'id':"0B-fDiFA73MH_N1ZCNVNYcW0tRFk"}]}
            res = DRIVE.files().insert(convert=convert, body=metadata,
                    media_body=path2file, fields='mimeType,exportLinks').execute()
            if res:
                print('Uploaded "%s" (%s)' % (filename, res['id']))
        return True
    except:
        return False
