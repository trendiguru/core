__author__ = 'jeremy'

import os

import Utils


def make_index(dir):
    '''
    makes index.html based on  files in directory
    '''
    print('dir:' + str(dir))
    files = Utils.files_in_directory(dir)
    print('files:')
    print(files)
    htmlfiles = []
    for file in files:
        if file.endswith('html'):
            htmlfiles.append(file)
    htmlfiles.sort(key=lambda x: os.path.getmtime(x))
    write_index_html(dir, htmlfiles)
    print('htmlfiles:')
    print(htmlfiles)

def write_index_html(dir, files):
    f = open('index.html', 'w')
    # write html file
    f.write('<HTML><HEAD><TITLE>classifier, fingerprint results</TITLE>\n')
    # <a href="http://www.w3schools.com">Visit W3Schools</a>
    for file in files:
        f.write('<br>\n')
        f.write('<a href=\"' + str(file) + '\">' + str(file) + ' <\\a>\n')

    f.write('</html>\n')
    f.close


if __name__ == "__main__":
    print('start')
    make_index('classifier_results')
