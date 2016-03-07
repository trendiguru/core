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
        children = service.children().list(folderId=folder_id, q="name contains ebay" ).execute()
        for child in children.get('items', []):
            c_id = child['id']
            print ('File Id: %s' % c_id)
            break
    except errors.HttpError, error:
        if error.resp.status != 404:
            print ('An error occurred: %s' % error)
        return False

    return True, c_id

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
        file_exists, file_id = is_file_in_folder(DRIVE, folder_id=parent_folder, file_name='ebay')
        if file_exists:
            DRIVE.children().delete(folderId=parent_folder, childId=file_id).execute()

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
