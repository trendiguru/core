__author__ = 'jeremy'

import os

from trendi import Utils


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

def generate_html_allresults(orig,gt,nnbefore,nnafter,pdbefore,pdafter):
    f = open('index.html', 'w')
    # write html file
    f.write('<HTML><HEAD><TITLE>classifier, fingerprint results</TITLE></HEAD>\n')
    # <a href="http://www.w3schools.com">Visit W3Schools</a>
    origfiles=[af for af in os.listdir(orig) if '.jpg' in af]
    for a_file in origfiles:
        f.write('<br>\n')
        origfile = os.path.join(orig,a_file)
        gtfile = os.path.join(gt,a_file[:-4]+'.png_legend.jpg')
        nnb4file = os.path.join(nnbefore,a_file[:-4]+'_legend.jpg')
        nnafterfile = os.path.join(nnafter,a_file[:-4]+'_nnconclusions_legend.jpg')
        pdb4file = os.path.join(pdbefore,a_file[:-4]+'_pdparse_legend.jpg')
        pdafterfile = os.path.join(pdafter,a_file[:-4]+'_pdconclusions_legend.jpg')
        line = 'orig<img height="400" src="'+origfile+'">'
        f.write(line)
        line = 'gt<img height="400" src="'+gtfile+'">'
        f.write(line)
        line = 'nnb4<img height="400" src="'+nnb4file+'">'
        f.write(line)
        line = 'nnafter<img height="400" src="'+nnafterfile+'">'
        f.write(line)
        line = 'pdb4<img height="400" src="'+pdb4file+'">'
        f.write(line)
        line = 'pdafterfile<img height="400" src="'+pdafterfile+'">'
        f.write(line)

    f.write('</html>\n')
    f.close

if __name__ == "__main__":
    print('start')
#    make_index('classifier_results')

    origdir = '/home/jeremy/image_dbs/colorful_fashion_parsing_data/images/test/'
    gt = '/home/jeremy/image_dbs/colorful_fashion_parsing_data/output/groundtruth'
    nnb4 = '/home/jeremy/image_dbs/colorful_fashion_parsing_data/output/150x100_nn2_output_010516'
    nnafter = '/home/jeremy/image_dbs/colorful_fashion_parsing_data/output/150x100_nn2_output_010516_afterconclusions'
    pdb4 = '/home/jeremy/image_dbs/colorful_fashion_parsing_data/output/pd'
    pdafter = '/home/jeremy/image_dbs/colorful_fashion_parsing_data/output/pd'
    generate_html_allresults(origdir,gt,nnb4,nnafter,pdb4,pdafter)