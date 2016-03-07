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
ebay_id = '1JMRyLEf4jeEIH7Af07brOQStLxK-o4Bquho0JlsHttE'
parent_folder = "0B-fDiFA73MH_N1ZCNVNYcW0tRFk"

def is_file_in_folder(service, folder_id, file_name):
    # param={"name":='ebay'"}
    try:
        query_by_name = "trashed = false and fullText contains " + file_name
        children = service.children().list(folderId=folder_id, q=query_by_name).execute()
        childs =children.get('item')
        print (childs)
        if len(childs)<1:
            return False, []
        for c in childs:
            child_id = c['id']
            service.children().delete(folderId=parent_folder, childId=child_id).execute()
    except errors.HttpError, error:
        if error.resp.status != 404:
            print ('An error occurred: %s' % error)


def upload2drive(FILE2INSERT):
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

        is_file_in_folder(DRIVE, folder_id=parent_folder, file_name="'ebay'")

        filename,path2file,convert = FILE2INSERT
        metadata = {'title': filename,
                    'parents':[{'id': parent_folder}]}
        res = DRIVE.files().insert(convert=convert, body=metadata,
                media_body=path2file, fields='id').execute()
        if res:
            print('Created new file named : "%s"  file id: %s' % (filename, res['id']))
        return True
    except:
        return False
