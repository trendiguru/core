__author__ = 'jeremy'

import Utils


def make_index(dir):
    '''
    makes index.html based on  files in directory
    '''
    files = Utils.files_in_directory(dir)
    htmlfiles = []
    for file in files:
        if file.endswith('html'):
            htmlfiles.append
    write_index_html(dir, files)


def write_index_html(dir, files):
    f = open('index.html', 'w')
    # write html file
    f.write('<HTML><HEAD><TITLE>classifier, fingerprint results</TITLE>\n')
    # <a href="http://www.w3schools.com">Visit W3Schools</a>
    for file in files:
        f.write('<br>\n')
        f.write('<a href=\"' + str(file) + '\">' + str(file) + '<\\a>\n')

    f.write('</html>\n')
    f.close


if __name__ == "__main__":
    print('start')
    make_index('classifier_results')
