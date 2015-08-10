
import matlab.engine
import subprocess
import urllib
import sys
import shutil
import requests
import pd
import cv2
import numpy

pictures = ['http://www.eonline.com/eol_images/Entire_Site/2014030/rs_634x1024-140130054607-634-Halle-Berry-JR-13014.jpg',
'http://static.gofugyourself.com/uploads/2014/08/halle-berry-emmys-2014-454167512.jpg',
'http://media2.popsugar-assets.com/files/2011/05/21/4/192/1922564/764b33c40973e58d_ActressHa_Dimit_64902619_Max.xxxlarge/i/Halle-Berry-Sexy-Black-Halston-Dress-Fifi-Awards.jpg',
'http://cdn-static.farfetch.com/portalcontent/static/images/_editorial/from-catwalk-to-celluloid/costume%20designers%20the%20fifth%20element.jpg',
'https://cbsla.files.wordpress.com/2012/02/140009389.jpg',
'http://www.redcarpet-fashionawards.com/wp-content/uploads/2011/01/angelina-jolie-versace.jpg',
'http://cdn.images.express.co.uk/img/dynamic/79/590x/secondary/Angelina-Jolie-Unbroken-premiere-222787.jpg',
'http://www.usmagazine.com/uploads/assets/photo_galleries/regular_galleries/2710-critics-choice-awards-2015-red-carpet-fashion-stars/photos/1421373480_angelina-jolie-zoom.jpg',
'http://i.dailymail.co.uk/i/pix/2013/05/12/article-0-19BE5726000005DC-677_964x1385.jpg',
'http://talkingmakeup.com/pics/news3/miller4.jpg',
'http://i.huffpost.com/gen/1899905/thumbs/o-PENELOPE-CRUZ-900.jpg?16',
'http://www.openingceremony.us/userfiles/image/news/sept11/092811-fine-wine-brigitte-bardot/092811-fine-wine-brigitte-bardot64.jpg',
'http://www.whatgoesaroundnyc.com/blog/wp-content/uploads/2013/07/brigitte-bardot-1.jpg',
'http://nowmagazine.media.ipcdigital.co.uk/11140/00001de7c/8248_orh480w360/Kate-Moss-full-body.jpg',
'http://www.standard.co.uk/news/article6829119.ece/binary/original/Kate%20Moss.jpg',
'http://fc08.deviantart.net/fs70/i/2013/113/d/b/anna_kournikova_walks_by_lowerrider-d62q4db.jpg',
'http://3.bp.blogspot.com/-JA5Q5AVFg3Q/VANHoL_YffI/AAAAAAAAAwc/nMm3l9OBuLw/s1600/anna%2Bkournikova.jpg',
'http://www.eonline.com/eol_images/Entire_Site/201458/rs_634x1024-140608104926-634.Anna-Kournikova-jmd-060814.jpg',
'http://iv1.lisimg.com/image/6675539/600full-anna-kournikova.jpg',
'http://4.bp.blogspot.com/-ITSNsF1IdXg/Utn6rx4sufI/AAAAAAAAAnk/YQG9lbVeGns/s1600/Anna+Kournikova+Picture+2014+08.jpg',
'http://i.perezhilton.com/wp-content/uploads/2014/03/jennifer-connelly-louis-vuitton-noah-nyc-premiere__oPt.jpg',
'http://www.eonline.com/eol_images/Entire_Site/2014211/rs_634x1024-140311102304-634-jennifer-connelly-mexico-hard-candy-dress-031114.jpg',
'http://oud.girlscene.nl/images/library/articles/images01/girlscene_nieuw/doutzen-kroez-lucky-one-celebutopia-CD041903.jpg',
'http://i.dailymail.co.uk/i/pix/2014/10/21/1413888774966_wps_4_BEVERLY_HILLS_CA_OCTOBER_.jpg',
'http://www.gotceleb.com/wp-content/uploads/celebrities/winona-ryder/red-magazine-april-2014/Winona-Ryder:-Red-Magazine--05.jpg',
'http://watchyourstar.com/wp-content/uploads/2015/07/Charlize-Theron-Very-Sexy-Photo-in-Cool-Black-Dress.jpg',
'http://i1.cdnds.net/13/09/300x450/charlize_theron.jpg',
'http://www.celebritywallpapers.ws/Celebrities/Charlize_Theron/Charlize-Theron-002.jpg',
'http://www.onlinepromdress.com/upfile/Celebrity%20Dresses/Hot%20Selling%20Celebrity%20Dresses/Charlize%20Theron%20Ruched%20Dress%20for%20Venice%20Film%20Festival%202008%20Red%20Carpet%20with%20Straps.jpg',
'http://hd.maltebauer.de/thefifthelement_bd-vs-bdremastered/03_bd.png',
'http://www.breakmystyle.com/wp-content/uploads/blogger/_r4vVgUERWxE/TTBK73PVojI/AAAAAAAAA6Q/gkDr2WSaRV4/s1600/kate_moss_1.jpg']


#results=[]

#for pic in pictures:

#    results.append(pd.get_parse_mask(pic))
results=[]


#for pic in pictures:

#    results.append(pd.get_parse_mask(image_url=pic))
#    mask,labels,pose = pd.get_parse_mask(image_url=pic)
#    results.append(mask)

def downloadUrls():

        for pic in pictures:

            stripped_name=pic.split('//')[1]
            modified_name=stripped_name.replace('/','_')

            print(modified_name)
            response = requests.get(pic, stream=True)
            with open(modified_name, 'wb') as out_file:
                 shutil.copyfileobj(response.raw, out_file)
            del response
    

def htmlRes():

    with open('getUrlResults.html', 'w') as outfile:
        outfile.write('<table style="height:228px">') #style="width:304px;height:228px;"

        for pic in pictures:

            stripped_name=pic.split('//')[1]
            modified_name=stripped_name.replace('/','_')
            modified_name=modified_name.replace('\'','\"')

            outfile.write('<tr>')
	    stylestring="height:300px;" 
            mystring='<td><img src='+modified_name+' style='+'\"'+ stylestring+'\" ></td>\n' 
            outfile.write(mystring)

            png_name=modified_name.replace('.jpg','.png')
            show_parse(png_name)
            mystring1='<td><img src='+'Col'+png_name+' style='+'\"' + stylestring+'\" ></td>\n'
            outfile.write(mystring1)
            outfile.write('</tr>')

        outfile.write('</table>')
        outfile.close()

def show_parse(filename, img_array=None):

    if filename is not None:
        img_array = cv2.imread(filename)
    if img_array is not None:
        # minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(img_array)
#        maxVal = numpy.amax(img_array)
        maxVal = 56 #this is number of pd classes
        scaled = numpy.multiply(img_array, int(1000/ maxVal))
        dest = cv2.applyColorMap(scaled, cv2.COLORMAP_RAINBOW)
#        cv2.imshow("dest", dest)
        Col_name='Col'+filename
        cv2.imwrite(Col_name, dest)
#       cv2.imshow('pd' ,dest)
#        cv2.waitKey(100)
    else:
	print('oops, file '+filename+' not found')


if __name__ == '__main__':
    htmlRes()


