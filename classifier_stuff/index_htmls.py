__author__ = 'jeremy'

import os
import time

from trendi import Utils

def make_indices_onedeep(dir):
    #do current direcotry

    print('indexing directory '+str(dir))
    make_index(dir)
    #do subdirectories
    dirs = [os.path.join(dir,d) for d in os.listdir(dir) if os.path.isdir(os.path.join(dir,d))]
    dirs=sorted(dirs,key=os.path.getmtime,reverse=True)  #sort by date
#    dirs.sort()

    print('top dirs in '+dir+':'+str(dirs))
    for d in dirs:
        print('onedeep now making index.html for '+str(d))
        make_index(d)

def make_index(dir):
    '''
    makes index.html linking all html files in directory
    '''
    print('make_index now making index for dir:' + str(dir))
#    files = Utils.files_in_directory(dir)
#    files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir,f)) ]

    sortedfiles=sorted([os.path.join(dir,f) for f in os.listdir(dir) if os.path.isfile(os.path.join(dir,f)) ],
                 key=os.path.getmtime,reverse=True)
    files=[os.path.basename(f) for f in sortedfiles]
    files.sort() #dont sort by date, it mixes nets up
    print('files in:'+str(files))
#    print(files)
    #sort by time
    sorteddirs=sorted([os.path.join(dir,f) for f in os.listdir(dir) if os.path.isdir(os.path.join(dir,f)) ],
                 key=os.path.getmtime,reverse=True)
    sorteddirs.sort()  #undo the sort by time
    dirs = [os.path.basename(d) for d in sorteddirs ]
    print('dirs in '+str(dir)+':'+str(dirs))
#    dirs = [f for f in os.listdir(dir) if os.path.isdir(os.path.join(dir,f)) ]
    dirs.sort() #dont sort by date, it mixes nets up
    htmlfiles = []
    for file in files:
#        actual_path =
        if file=='index.html':
            continue
        htmlfiles.append(file)
#        if file.endswith('html'):
#            htmlfiles.append(file)
#        if os.path.isdir(file):
#            htmlfiles.append(file)

#   sort by mod time
#    htmlfiles.sort(key=lambda x: os.path.getmtime(os.path.join(dir,x)))
    #sort alphabetically
#    htmlfiles.sort()
    for d in dirs:
        htmlfiles.append(d)
    print('files+dirs in:'+str(dir))
    print(htmlfiles)
    write_index_html_with_images(dir, htmlfiles)
    print('wrote index.html for files in dir:' +str(dir))
 #   print(htmlfiles)

def write_index_html(dir, files):
    '''makes a page with links to all files in dir
    '''
    indexname = os.path.join(dir,'index.html')
    print('writing to '+str(indexname))
    f = open(indexname, 'w')
    # write html file
    f.write('<HTML><HEAD><TITLE>Results</TITLE>\n')
    # <a href="http://www.w3schools.com">Visit W3Schools</a>
    for file in files:
        fullpath = os.path.join(dir,file)
        modtime = time.ctime(os.path.getmtime(fullpath))
        f.write('<br>\n')
        f.write('<a href=\"' + str(file) + '\">' + str(file) + ' </a> ' + modtime+'\n')

    f.write('</html>\n')
    f.close

def write_index_html_with_images(dir, files):
    '''makes a page with image links to all files in dir
    '''
    indexname = os.path.join(dir,'index.html')
    print('writing to '+str(indexname))
    f = open(indexname, 'w')
    # write html file
    f.write('<HTML><HEAD><TITLE>classifier, fingerprint results</TITLE>\n')
    # <a href="http://www.w3schools.com">Visit W3Schools</a>
    for file in files:
        f.write('<br>\n')

        fullpath = os.path.join(dir,file)
        modtime = time.ctime(os.path.getmtime(fullpath))

       # f.write('<a href=\"' + str(file) + '\">' + str(file) + ' <\\a>\n')
        print('writing line for file:'+file)
        if file[-4:] == '.jpg' or file[-4:] == '.png':
            print('jpg line for '+file)
            f.write('<a href=\"'+str(file)+'\">'+str(file)+'<img src = \"'+file+'\" style=\"width:300px\"></a> ' + modtime+'\n')
        else:
            print('nonjpg line for '+file)
            f.write('<a href=\"' + str(file) + '\">' + str(file) + ' </a> ' + modtime+'\n')

    f.write('</html>\n')
    f.close

def generate_filtered_index_html(dir, filter=''):
    '''makes a page with image links to all files in dir with filter in the filename
    '''
#    files = [os.path.join(dir,f) for f in os.listdir(dir) if filter in f]
# dont include dir - make files relative so html can be portable
    files = [f for f in os.listdir(dir) if filter in f]
    files.sort()
    write_index_html_with_images(dir,files)


def generate_html_allresults(orig,gt,nnbefore,nnafter,pdbefore,pdafter):
    f = open('index.html', 'w')
    # write html file
    f.write('<HTML><HEAD><TITLE>classifier, fingerprint results</TITLE></HEAD>\n')
    # <a href="http://www.w3schools.com">Visit W3Schools</a>
    origfiles=[af for af in os.listdir(orig) if '.jpg' in af]
    for a_file in origfiles:
        f.write('<br>\n')
        origfile = os.path.join(orig,a_file)
        origfile = origfile[1:]  #remove initial / and use link since html cannot reference abs path
        print('making html for file:'+a_file)
        gtfile = os.path.join(gt,a_file[:-4]+'.png_legend.jpg')
        gtfile = gtfile[1:]  #remove initial / and use link since html cannot reference abs path
        nnb4file = os.path.join(nnbefore,a_file[:-4]+'_legend.jpg')
        nnb4file = nnb4file[1:]
        nnafterfile = os.path.join(nnafter,a_file[:-4]+'_nnconclusions_legend.jpg')
        nnafterfile = nnafterfile[1:]
        pdb4file = os.path.join(pdbefore,a_file[:-4]+'_pdparse_legend.jpg')
        pdb4file = pdb4file[1:]
        pdafterfile = os.path.join(pdafter,a_file[:-4]+'_pdconclusions_legend.jpg')
        pdafterfile = pdafterfile[1:]
        line = 'orig<img height="400" src="'+origfile+'">\n'
        print line
        f.write(line)
        line = 'gt<img height="400" src="'+gtfile+'">\n'
        print line
        f.write(line)
        line = 'nnb4<img height="400" src="'+nnb4file+'">\n'
        print line
        f.write(line)
        line = 'nnaft<img height="400" src="'+nnafterfile+'">\n'
        print line
        f.write(line)
        line = 'pdb4<img height="400" src="'+pdb4file+'">\n'
        print line
        f.write(line)
        line = 'pdaft<img height="400" src="'+pdafterfile+'">\n'
        print line
        f.write(line)

    f.write('</html>\n')
    f.close

if __name__ == "__main__":
    print('start')
#    make_index('classifier_results')
    make_indices_onedeep('/var/www/results')

    if(0):
        origdir = '/home/jeremy/image_dbs/colorful_fashion_parsing_data/images/test/'
        gt = '/home/jeremy/image_dbs/colorful_fashion_parsing_data/output/groundtruth'
        nnb4 = '/home/jeremy/image_dbs/colorful_fashion_parsing_data/output/150x100_nn2_output_010516'
        nnafter = '/home/jeremy/image_dbs/colorful_fashion_parsing_data/output/150x100_nn2_output_010516_afterconclusions'
        pdb4 = '/home/jeremy/image_dbs/colorful_fashion_parsing_data/output/pd'
        pdafter = '/home/jeremy/image_dbs/colorful_fashion_parsing_data/output/pd'
        generate_html_allresults(origdir,gt,nnb4,nnafter,pdb4,pdafter)


        origdir = '/home/jeremy/image_dbs/colorful_fashion_parsing_data/images/test/'
        gt = '/home/jeremy/image_dbs/colorful_fashion_parsing_data/output/groundtruth'
        nnb4 = '/home/jeremy/image_dbs/colorful_fashion_parsing_data/output/600x400_nn1_output_010516'
        nnafter = '/home/jeremy/image_dbs/colorful_fashion_parsing_data/output/600x400_nn1_output_010516_afterconclusions'
        pdb4 = '/home/jeremy/image_dbs/colorful_fashion_parsing_data/output/pd'
        pdafter = '/home/jeremy/image_dbs/colorful_fashion_parsing_data/output/pd'
        generate_html_allresults(origdir,gt,nnb4,nnafter,pdb4,pdafter)

