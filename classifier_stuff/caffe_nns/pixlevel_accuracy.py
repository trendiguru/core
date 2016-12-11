__author__ = 'jeremy' #ripped from shelhamer pixlevel iou code at caffe home

import os
import numpy as np
import caffe # If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.
import cv2
import argparse
import logging
logging.basicConfig(level=logging.INFO)
import time


from trendi import constants
from trendi.utils import imutils
from trendi import Utils
from trendi.classifier_stuff.caffe_nns import caffe_utils
from trendi.classifier_stuff.caffe_nns import jrinfer
from trendi.paperdoll import neurodoll_falcon_client as nfc
from trendi import kassper
from trendi.paperdoll import paperdoll_parse_enqueue
from trendi import pipeline

def open_html(htmlname,model_base,solverproto,classes,results_dict):
    netname = caffe_utils.get_netname(solverproto)
    with open(htmlname,'a') as g:
        g.write('<!DOCTYPE html>')
        g.write('<html>')
        g.write('<head>')
        g.write('<title>')
        dt=datetime.datetime.today()
        g.write(model_base+' '+dt.isoformat())
        g.write('</title>')
        g.write('</head>')
        g.write('<body>')
#        g.write('categories: '+str(constants.web_tool_categories)+'<br>'+'\n')
        g.write('<br>\n')
        g.write('pixlevel results generated on '+ str(dt.isoformat()))
        g.write('<br>\n')
        g.write('solver:'+solverproto+'\n<br>')
        g.write('model:'+model_base+'\n'+'<br>')
        g.write('netname:'+netname+'\n<br>')
        g.write('iter:'+str(results_dict['iter'])+' loss:'+str(results_dict['loss'])+'\n<br>')
        g.write('overall acc:'+str(results_dict['overall_acc'])+' mean acc:'+str(results_dict['mean_acc'])+
                ' fwavac:'+str(results_dict['fwavacc'])+'\n<br>')
        g.write('mean iou:'+str(results_dict['mean_iou'])+'\n<br>')

        g.write('<table style=\"width:100%\">\n')
        g.write('<tr>\n')
        g.write('<th align="left">')
        g.write('metric')
        g.write('</th>\n')
        g.write('<th align="left">')
        g.write('fw avg.')
        g.write('</th>\n')
        for i in range(len(classes)):
            g.write('<th align="left">')
            g.write(classes[i])
            g.write('</th>\n')
        g.write('</tr>\n')

def close_html(htmlname):
    with open(htmlname,'a') as g:
        g.write('</table><br>')
        plotfilename = 'imagename.png'
        g.write('<a href=\"'+plotfilename+'\">plot<img src = \"'+plotfilename+'\" style=\"width:300px\"></a>')
        g.write('</body>')
        g.write('</html>')

def write_html(htmlname,results_dict):
#    results_dict = {'iter':iter,'loss':loss,'class_accuracy':acc.tolist(),'overall_acc':overall_acc.tolist(),'mean_acc':mean_acc.tolist(),'class_iou':iu.tolist(),'mean_iou':mean_iou.tolist(),'fwavacc':fwavacc.tolist()}
    with open(htmlname,'a') as g:
        #write class accuracies
        g.write('<tr>\n')
        g.write('<td>')
        g.write('accuracy')
        g.write('</td>\n')
        g.write('<td>')
        g.write(str(round(results_dict['fwavacc'],3)))
        g.write('</td>\n')
        for i in range(len(results_dict['class_accuracy'])):
            g.write('<td>')
            class_accuracy = results_dict['class_accuracy'][i]
            g.write(str(round(class_accuracy,3)))
            g.write('</td>\n')
        g.write('</tr>\n<br>\n')

        #write class iou
        g.write('<tr>\n')
        g.write('<td>')
        g.write('iou')
        g.write('</td>\n')
        g.write('<td>')
        g.write(str(round(results_dict['mean_iou'],3)))  #the mean iou might not be same as fwiou which is what should go here
        g.write('</td>\n')
        for i in range(len(results_dict['class_iou'])):
            g.write('<td>')
            class_iou = results_dict['class_iou'][i]
            g.write(str(round(class_iou,3)))
            g.write('</td>\n')
        g.write('</tr>\n<br>\n')

def write_textfile(caffemodel, solverproto, threshold,model_base,dir=None,classes=None):
    if dir is None:
        protoname = solverproto.replace('.prototxt','')
        dir = 'multilabel_results-'+protoname+'_'+model_base.replace('.caffemodel','')
    Utils.ensure_dir(dir)
    fname = os.path.join(dir,model_base+'results.txt')
    with open(fname,'a') as f:
        f.write(model_base+' threshold = '+str(threshold)+'\n')
        f.write('solver:'+solverproto+'\n')
        f.write('model:'+caffemodel+'\n')
        f.write('categories: '+str(classes)+ '\n')
        f.close()

def do_pixlevel_accuracy(caffemodel,n_tests,layer,classes=constants.ultimate_21,testproto=None,solverproto=None, iter=0, savepics=True):
#to do accuracy we prob dont need to load solver
    caffemodel_base = os.path.basename(caffemodel)
    dir = 'pixlevel_results-'+caffemodel_base.replace('.caffemodel','')
    Utils.ensure_dir(dir)
    if savepics:
        picsdir = os.path.join(dir,'pics')
        Utils.ensure_dir(picsdir)
    else:
        picsdir = False
    htmlname = os.path.join(dir,dir+'.html')
    detailed_outputname = htmlname[:-5]+'.txt'
    print('saving net of {} {} to dir {} and file {}'.format(caffemodel,solverproto,htmlname,detailed_outputname))

    val = range(n_tests)
    if args.gpu:
        caffe.set_mode_gpu()
        caffe.set_device(int(args.gpu))
    else:
        caffe.set_mode_cpu()

    if(solverproto is not None): #do this the old way with sgdsolver
        solver = caffe.SGDSolver(solverproto)
        solver.net.copy_from(caffemodel)
        print('using net defined by {} and {} '.format(solverproto,caffemodel))
        answer_dict = jrinfer.seg_tests(solver, picsdir, val, layer=layer,outfilename=detailed_outputname)

    elif(testproto is not None):  #try using net without sgdsolver
        net = caffe.Net(testproto,caffemodel, caffe.TEST)
        answer_dict = jrinfer.do_seg_tests(net, iter, picsdir, val, layer=layer, gt='label',outfilename=detailed_outputname)



  #   in_ = np.array(im, dtype=np.float32)
  #   net.blobs['data'].reshape(1, *in_.shape)
  #   net.blobs['data'].data[...] = in_
  #   # run net and take argmax for prediction
  #   net.forward()
  #   out = net.blobs['seg-score'].data[0].argmax(axis=0)

    open_html(htmlname,caffemodel,solverproto,classes,answer_dict)
    write_html(htmlname,answer_dict)
    close_html(htmlname)

def get_pixlevel_confmat_using_falcon(images_and_labels_file,labels=constants.ultimate_21, save_dir='./nd_output'):
    with open(images_and_labels_file,'r') as fp:
        lines = fp.readlines()
    n_cl = len(labels)
    hist = np.zeros((n_cl, n_cl))
    loss = 0
    print('n channels: '+str(n_cl))
    start_time=time.time()
    for line in lines:
        imagefile = line.split()[0]
        gtfile = line.split()[1]
        img_arr = cv2.imread(imagefile)
        if img_arr is None:
            logging.warning('could not get image data from '+imagefile)
            continue
        gt_data = cv2.imread(gtfile)
        if gt_data is None:
            logging.warning('could not get gt data from '+gtfile)
            continue
        if len(gt_data.shape) == 3:
            logging.warning('got 3 chan image for mask, taking chan 0 '+gtfile)
            gt_data = gt_data[:,:,0]

        dic = nfc.pd(img_arr)
        if not dic['success']:
            logging.debug('nfc pd not a success')
            continue
        net_data = dic['mask']
        print('sizes of gt {} net output {}'.format(gt_data.shape,net_data.shape))

        hist += jrinfer.fast_hist(gt_data.flatten(),net_data.flatten(),n_cl)

        if save_dir:
            Utils.ensure_dir(save_dir)
            gt_name=os.path.basename(imagefile)[:-4]+'_gt_legend.jpg'
            gt_name=os.path.join(save_dir,gt_name)
            ndout_name=os.path.basename(imagefile)[:-4]+'_ndout_legend.jpg'
            ndout_name=os.path.join(save_dir,ndout_name)
            imutils.show_mask_with_labels(gt_data,labels,original_image=imagefile,save_images=True,visual_output=False,savename=gt_name)
            imutils.show_mask_with_labels(net_data,labels,original_image=imagefile,save_images=True,visual_output=False,savename=ndout_name)        # compute the loss as well

    results_dict = jrinfer.results_from_hist(hist,save_file=os.path.join(save_dir,'output.html'))
    print results_dict
    elapsed_time=time.time()-start_time
    print('elapsed time: '+str(elapsed_time)+' tpi:'+str(float(elapsed_time)/len(lines)))
    return hist

def create_swimsuit_mask_using_grabcut_only(dir,bathingsuit_index,labels=constants.pixlevel_categories_v2,skinlayer = 45):
    '''
    :param dir: directory of images for which to generate masks
    :param category_index: category from pixlevel v2
    :param skinlayer - the index of the layer for skin which is 45 in pixlevel_categories_v2
    :return: create mask files  file.png , also convert to webtool style (index in red channel)
    27 mens swimwear
    19 bikini
    20 womens_nonbikini
    '''
    print('creating masks for swimsuits category {} label {} skincat {} label {}'.format(bathingsuit_index,labels[bathingsuit_index],skinlayer,labels[skinlayer]))
    files=[os.path.join(dir,f) for f in os.listdir(dir) if not 'legend' in f]
    print(str(len(files))+' files to make into masks '+dir)
    for f in files:
        img_arr = cv2.imread(f)
        print('file '+f + ' shape '+str(img_arr.shape))
        h,w = img_arr.shape[0:2]
        if img_arr is None:
            continue
        dic = nfc.pd(img_arr)
        if not dic['success']:
            logging.debug('nfc pd not a success')
            continue
        mask = dic['mask']
        logging.debug('sizes of gt {}'.format(mask.shape))
        background = np.array((mask==0)*1,dtype=np.uint8)
        foreground = np.array((mask>0)*1,dtype=np.uint8)
        if(0):  #use nd skin layer
            skin= np.array((mask==skinlayer)*1,dtype=np.uint8)
            bathingsuit=np.array((mask!=0)*1,dtype=np.uint8) *  np.array((mask!=skinlayer)*bathingsuit_index,dtype=np.uint8)
        else: #use nadav skindetector
            #skin = kassper.skin_detection_with_grabcut(img_arr, img_arr, face=None, skin_or_clothes='skin')
            skin =  kassper.skin_detection(img_arr)
            nonskin = np.array(skin==0,dtype=np.uint8)
            bathingsuit=np.multiply(foreground, nonskin) *bathingsuit_index
            print('vals in bathingsuit '+str(np.unique(bathingsuit)))

#        out_arr = skin + nonskin*
        n_bg_pixels = np.count_nonzero(background)
        n_fg_pixels = np.count_nonzero(foreground)
        n_skin = np.count_nonzero(skin)
        n_bathingsuit = np.count_nonzero(bathingsuit)
        logging.debug('size  of fg {} pixels {} bg {} pixels {} bathings {} {}'.format(foreground.shape,n_fg_pixels,background.shape,n_bg_pixels,bathingsuit.shape,n_bathingsuit))
        outfile = os.path.join(dir,os.path.basename(f)[:-4]+'.png')
        logging.info('writing bathingsuitmask to '+outfile)
        cv2.imwrite(outfile,bathingsuit)
        #save new mask
        mask_legendname = f[:-4]+'_skin_nogc.jpg'
        imutils.show_mask_with_labels(outfile,labels=labels,original_image=f,save_images=True,savename=mask_legendname)
        #save original mask
        orig_legendname = f[:-4]+'_original_legend.jpg'
        imutils.show_mask_with_labels(mask,labels=constants.ultimate_21,original_image=f,save_images=True,savename=orig_legendname)
    convert_masks_to_webtool(dir)

def convert_masks_to_webtool(dir,suffix_to_convert_from='.png',suffix_to_convert_to='_webtool.png'):
    '''
    images saved as .bmp seem to have a single grayscale channel, and an alpha.
    using 'convert' to convert those to .png doesn't help, same story. the web tool example images have the red channel
    as index, so this func converts to that format. actually i will try r=g=b=index, hopefully thats ok too - since that
    will be compatible with rest of our stuff...that didnt work , the tool really wants R=category, B=G=0
    '''
    files_to_convert=[os.path.join(dir,f) for f in os.listdir(dir) if suffix_to_convert_from in f and not suffix_to_convert_to in f]
    print(str(len(files_to_convert))+' files in '+dir)
    for f in files_to_convert:
        img_arr = cv2.imread(f)
        print('file '+f + ' shape '+str(img_arr.shape)+ ' uniques:'+str(np.unique(img_arr)))
        h,w = img_arr.shape[0:2]
        out_arr = np.zeros((h,w,3))
        out_arr[:,:,0] = 0  #B it would seem this can be replaced by out_arr[:,:,:]=img_arr, maybe :: is used here
        out_arr[:,:,1] = 0  #G
        out_arr[:,:,2] = img_arr[:,:,0]  #R

        newname = os.path.join(dir,os.path.basename(f).replace(suffix_to_convert_from,suffix_to_convert_to))
        print('outname '+str(newname))
        cv2.imwrite(newname,out_arr)
        return out_arr



if __name__ =="__main__":

    default_solverproto = '/home/jeremy/caffenets/multilabel/deep-residual-networks/prototxt/ResNet-101-test.prototxt'
    default_testproto = '/home/jeremy/caffenets/multilabel/deep-residual-networks/prototxt/ResNet-101-test.prototxt'
    default_caffemodel = '/home/jeremy/caffenets/production/multilabel_resnet101_sgd_iter_120000.caffemodel'

    parser = argparse.ArgumentParser(description='multilabel accuracy tester')
    parser.add_argument('--solverproto',  help='solver prototxt',default=None)
    parser.add_argument('--testproto',  help='val prototxt',default=None)
    parser.add_argument('--caffemodel', help='caffmodel',default = default_caffemodel)
    parser.add_argument('--gpu', help='gpu #',default=0)
    parser.add_argument('--output_layer_name', help='output layer name',default='score')
    parser.add_argument('--n_tests', help='number of examples to test',default=200)
    parser.add_argument('--classes', help='class labels',default=constants.ultimate_21)
    parser.add_argument('--iter', help='iter',default=0)
    parser.add_argument('--savepics', help='iter',default=True)

    args = parser.parse_args()
    print(args)
    gpu = int(args.gpu)
    outlayer = args.output_layer_name
    n_tests = int(args.n_tests)

    testfile = '/home/jeremy/image_dbs/colorful_fashion_parsing_data/images_and_labelsfile_test.txt'
    get_pixlevel_confmat_using_falcon(testfile,labels=constants.ultimate_21, save_dir='./nd_output')

#    caffe.set_mode_gpu()
#    caffe.set_device(gpu)
#    print('using net defined by valproto {} caffmodel  {} solverproto {}'.format(args.testproto,args.caffemodel,args.solverproto))
#    do_pixlevel_accuracy(args.caffemodel,n_tests,args.output_layer_name,args.classes,solverproto = args.solverproto, testproto=args.testproto,iter=int(args.iter),savepics=args.savepics)





