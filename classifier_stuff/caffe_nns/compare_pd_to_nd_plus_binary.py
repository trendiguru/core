__author__ = 'jeremy'

import cv2
import pymongo
import subprocess
import os
from time import sleep
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)
import json
import re
import os
import cv2
import numpy as np
import logging
import subprocess
import sys
import json
from trendi.classifier_stuff.caffe_nns import conversion_utils
import pdb

from trendi.paperdoll import pd_falcon_client
from trendi import constants#
from trendi import Utils
from trendi.utils import imutils
from trendi.paperdoll import hydra_tg_falcon_client
from trendi.paperdoll import neurodoll_falcon_client
from trendi.utils import imutils
from trendi import pipeline
from trendi.downloaders import label_conversions
from trendi.classifier_stuff.caffe_nns import multilabel_accuracy
#from trendi import neurodoll

def get_live_pd_results(image_file,save_dir='/data/jeremy/image_dbs/tg/pixlevel/pixlevel_fullsize_test_pd_results',
                        new_labels = constants.fashionista_categories_augmented):
    '''
    todo - use post instead of get to avoid url and send img_arr instead
    :param image_file:
    :param save_dir:
    :param new_labels:
    :return:
    '''
    #use the api - so first get the image onto the web , then aim the api at it
    copycmd = 'scp '+image_file+' root@104.155.22.95:/var/www/results/pd_test/'+os.path.basename(image_file)
    print('copying file to webserver:'+copycmd)
    subprocess.call(copycmd,shell=True)
    sleep(1) #give time for file to get to extremeli - maybe unecessary (if subprocess is synchronous)
    url = 'http://extremeli.trendi.guru/demo/results/pd_test/'+os.path.basename(image_file)
    resp = pd_falcon_client.pd(url)
    print('resp:'+str(resp))
    label_dict = resp['label_dict']
    mask = resp['mask']
    #label_dict = {fashionista_categories_augmented_zero_based[i]:i for i in range(len(fashionista_categories_augmented_zero_based))}
    print label_dict
    if len(mask.shape) == 3:
        mask = mask[:,:,0]
    logging.debug('bincount before conclusions:'+str(np.bincount(mask.flatten())))
    #see https://github.com/trendiguru/tg_storm/blob/master/src/bolts/person.#py, hopefully this is ok without the face
#    final_mask = pipeline.after_pd_conclusions(mask, label_dict, person['face'])
    final_mask = pipeline.after_pd_conclusions(mask, label_dict,None)
    logging.debug('uniques:'+str(np.unique(final_mask)))
    logging.debug('bincount after conclusions:'+str(np.bincount(final_mask.flatten())))
    converted_mask = label_conversions.convert_pd_output(final_mask, label_dict, new_labels=new_labels)
#could also have used   get_pd_results_on_db_for_webtool.convert_and_save_results
    logging.debug('bincount after conversion:'+str(np.bincount(converted_mask.flatten())))

    print('save dir:'+save_dir)
    image_base = os.path.basename(image_file)
    save_name = os.path.join(save_dir,image_base[:-4]+'_pd.bmp')
    res=cv2.imwrite(save_name,converted_mask)
    print('save result '+str(res)+ ' for file '+save_name)
    return converted_mask

def get_hydra_tg_results(image_file):
    copycmd = 'scp '+image_file+' root@104.155.22.95:/var/www/results/pd_test/'+os.path.basename(image_file)
    print('copying file to webserver:'+copycmd)
    subprocess.call(copycmd,shell=True)
    sleep(1) #give time for file to get to extremeli - maybe unecessary (if subprocess is synchronous)
    url = 'http://extremeli.trendi.guru/demo/results/pd_test/'+os.path.basename(image_file)
    resp = hydra_tg_falcon_client.hydra_tg(url,thresholds=None)
    print('resp:'+str(resp))
    if not 'data' in resp:
        logging.warning('expected "data" in response '+str(resp))
    data = resp['data']
    return data


def get_live_nd_results(image_file,save_dir='/data/jeremy/image_dbs/tg/pixlevel/pixlevel_fullsize_test_pd_results',
                        new_labels = constants.fashionista_categories_augmented):
    #use the api - so first get the image onto the web , then aim the api at it


    #label_dict = {fashionista_categories_augmented_zero_based[i]:i for i in range(len(fashionista_categories_augmented_zero_based))}
    hydra_multilabel_results = get_hydra_tg_results(image_file)
    print('hydra_ml results:'+str(hydra_multilabel_results))
    fashionista_results = label_conversions.hydra_results_to_fashionista(hydra_multilabel_results)
    print('converted results:'+str(fashionista_results))
    print fashionista_results
    img_arr = cv2.imread(image_file)
    nd_mask = neurodoll_falcon_client.nd(img_arr)
    if len(nd_mask.shape) == 3:
        mask = nd_mask[:,:,0]
    logging.debug('bincount before conclusions:'+str(np.bincount(mask.flatten())))
    #see https://github.com/trendiguru/tg_storm/blob/master/src/bolts/person.#py, hopefully this is ok without the face
#    final_mask = pipeline.after_pd_conclusions(mask, label_dict, person['face'])
    final_mask = pipeline.after_pd_conclusions(mask, label_dict,None)
    logging.debug('uniques:'+str(np.unique(final_mask)))
    logging.debug('bincount after conclusions:'+str(np.bincount(final_mask.flatten())))
    converted_mask = label_conversions.convert_pd_output(final_mask, label_dict, new_labels=new_labels)
#could also have used   get_pd_results_on_db_for_webtool.convert_and_save_results
    logging.debug('bincount after conversion:'+str(np.bincount(converted_mask.flatten())))

    print('save dir:'+save_dir)
    image_base = os.path.basename(image_file)
    save_name = os.path.join(save_dir,image_base[:-4]+'_pd.bmp')
    res=cv2.imwrite(save_name,converted_mask)
    print('save result '+str(res)+ ' for file '+save_name)
    return converted_mask

def all_pd_results(filedir='/data/jeremy/image_dbs/tg/pixlevel/pixlevel_fullsize_test',
                    labelsdir='/data/jeremy/image_dbs/tg/pixlevel/pixlevel_fullsize_labels_fashionista_augmented_categories',
                    save_dir='/data/jeremy/image_dbs/tg/pixlevel/pixlevel_fullsize_test_pd_results',
                    labels=constants.fashionista_categories_augmented):

    print('starting all_pd_results , filedir:{}\nlabelsdir:{}\nsave_dir:{}labels:\n{}'.format(filedir,labelsdir,save_dir,labels))
    n_cl = len(labels)
    accumulated_confmat = np.zeros((n_cl, n_cl))
    files_to_test = [os.path.join(filedir,f) for f in os.listdir(filedir) if '.jpg' in f]
    if len(files_to_test)==0:
        print('no files in '+str(filedir))
        return

  #  files_to_test = files_to_test[0:3]
    print(str(len(files_to_test))+' files to test')

    tp = np.zeros(len(labels))
    tn = np.zeros(len(labels))
    fp = np.zeros(len(labels))
    fn = np.zeros(len(labels))

    for f in files_to_test:
        print('getting pd result for '+f)
        pd_mask = get_live_pd_results(f)
        logging.debug('pd bincount:'+str(np.bincount(pd_mask.flatten())))
        gt_file = os.path.join(labelsdir,os.path.basename(f).replace('.jpg','.png'))
        gt_mask = get_saved_mask_results(gt_file)
        logging.debug('gt bincount:'+str(np.bincount(gt_mask.flatten())))

        #save and send pd output
        labels = constants.fashionista_categories_augmented
        image_base = os.path.basename(f)
        save_name = os.path.join(save_dir,image_base[:-4]+'_pd.bmp')
        print('saving pd to :'+save_name)
        imutils.show_mask_with_labels(save_name,labels=labels,save_images=True,original_image=f)
        copycmd = 'scp '+save_name.replace('.bmp','_legend.jpg')+' root@104.155.22.95:/var/www/results/pd_test/'
        subprocess.call(copycmd,shell=True)
        #save and send gt
        save_name = os.path.join(save_dir,image_base[:-4]+'_gt.bmp')
        print('saving gt to :'+save_name)
        imutils.show_mask_with_labels(save_name,labels=labels,save_images=True,original_image=f)
        copycmd = 'scp '+save_name.replace('.bmp','_legend.jpg')+' root@104.155.22.95:/var/www/results/pd_test/'
        subprocess.call(copycmd,shell=True)

        confmat = fast_hist(gt_mask.flatten(), pd_mask.flatten(), n_cl)
        accumulated_confmat += confmat
        logging.info(accumulated_confmat)

        #do image level comparison (are the correct categories there)
        imagelevel_gt = np.zeros(len(labels))
        imagelevel_gt[np.unique(gt_mask)]=1
        imagelevel_pd = np.zeros(len(labels))
        imagelevel_pd[np.unique(pd_mask)]=1
        print('gt:'+str(imagelevel_gt))  #turn to int since print as float takes 2 lines
        print('est:'+str(imagelevel_pd))
        tp,tn,fp,fn = multilabel_accuracy.update_confmat(imagelevel_gt,imagelevel_pd,tp,tn,fp,fn)
        print('tp {}\ntn {}\nfp {}\nfn {}'.format(tp,tn,fp,fn))


    results_dict = results_from_hist(accumulated_confmat)
    logging.debug(results_dict)
    textfile = os.path.join(save_dir,'pixlevel_results.txt')
    with open(textfile,'a') as file_pointer:
        json.dump(results_dict,file_pointer,indent=4)
        file_pointer.close()
    htmlname = os.path.join(save_dir,'pd_results.html')
    results_to_html(htmlname,results_dict)

    imagelevel_dict = {'tp':tp.tolist(),'tn':tn.tolist(),'fp':fp.tolist(),'fn':fn.tolist()}
    textfile = os.path.join(save_dir,'imagelevel_results.txt')
    with open(textfile,'a') as file_pointer:
        json.dump(imagelevel_dict,file_pointer,indent=4)
        file_pointer.close()



def all_nd_results(filedir='/data/jeremy/image_dbs/tg/pixlevel/pixlevel_fullsize_test',
                    labelsdir='/data/jeremy/image_dbs/tg/pixlevel/pixlevel_fullsize_labels_fashionista_augmented_categories',
                    save_dir='/data/jeremy/image_dbs/tg/pixlevel/pixlevel_fullsize_test_pd_results',
                    labels=constants.fashionista_categories_augmented):

    print('starting all_nd_results , filedir:{}\nlabelsdir:{}\nsave_dir:{}labels:\n{}'.format(filedir,labelsdir,save_dir,labels))
    n_cl = len(labels)
    accumulated_confmat = np.zeros((n_cl, n_cl))
    files_to_test = [os.path.join(filedir,f) for f in os.listdir(filedir) if '.jpg' in f]
    if len(files_to_test)==0:
        print('no files in '+str(filedir))
        return

   # files_to_test = files_to_test[0:3]
    print(str(len(files_to_test))+' files to test')

    tp = np.zeros(len(labels))
    tn = np.zeros(len(labels))
    fp = np.zeros(len(labels))
    fn = np.zeros(len(labels))

    for f in files_to_test:
        print('getting pd result for '+f)
        nd_mask = get_live_nd_results(f)
        logging.debug('pd bincount:'+str(np.bincount(nd_mask.flatten())))
        gt_file = os.path.join(labelsdir,os.path.basename(f).replace('.jpg','.png'))
        gt_mask = get_saved_mask_results(gt_file)
        logging.debug('gt bincount:'+str(np.bincount(gt_mask.flatten())))

        #save and send output
        labels = constants.fashionista_categories_augmented
        image_base = os.path.basename(f)
        save_name = os.path.join(save_dir,image_base[:-4]+'_nd.bmp')
        print('saving nd to :'+save_name)
        imutils.show_mask_with_labels(save_name,labels=labels,save_images=True,original_image=f)
        copycmd = 'scp '+save_name.replace('.bmp','_legend.jpg')+' root@104.155.22.95:/var/www/results/pd_test/'
        subprocess.call(copycmd,shell=True)
        # #save and send gt - already sent by p d
        # save_name = os.path.join(save_dir,image_base[:-4]+'_gt.bmp')
        # print('saving gt to :'+save_name)
        # imutils.show_mask_with_labels(save_name,labels=labels,save_images=True,original_image=f)
        # copycmd = 'scp '+save_name.replace('.bmp','_legend.jpg')+' root@104.155.22.95:/var/www/results/pd_test/'
        # subprocess.call(copycmd,shell=True)

        confmat = fast_hist(gt_mask.flatten(), nd_mask.flatten(), n_cl)
        accumulated_confmat += confmat
        logging.info(accumulated_confmat)

        #do image level comparison (are the correct categories there)
        imagelevel_gt = np.zeros(len(labels))
        imagelevel_gt[np.unique(gt_mask)]=1
        imagelevel_pd = np.zeros(len(labels))
        imagelevel_pd[np.unique(nd_mask)]=1
        print('gt:'+str(imagelevel_gt))  #turn to int since print as float takes 2 lines
        print('est:'+str(imagelevel_pd))
        tp,tn,fp,fn = multilabel_accuracy.update_confmat(imagelevel_gt,imagelevel_pd,tp,tn,fp,fn)
        print('tp {}\ntn {}\nfp {}\nfn {}'.format(tp,tn,fp,fn))


    results_dict = results_from_hist(accumulated_confmat)
    logging.debug(results_dict)
    textfile = os.path.join(save_dir,'pixlevel_results.txt')
    with open(textfile,'a') as file_pointer:
        json.dump(results_dict,file_pointer,indent=4)
        file_pointer.close()
    htmlname = os.path.join(save_dir,'pd_results.html')
    results_to_html(htmlname,results_dict)

    imagelevel_dict = {'tp':tp.tolist(),'tn':tn.tolist(),'fp':fp.tolist(),'fn':fn.tolist()}
    textfile = os.path.join(save_dir,'imagelevel_results.txt')
    with open(textfile,'a') as file_pointer:
        json.dump(imagelevel_dict,file_pointer,indent=4)
        file_pointer.close()


def get_saved_mask_results(mask_file):
    img_arr = cv2.imread(mask_file)
    if  img_arr is None:
        print('couldnt open '+mask_file)
        return None
    if len(img_arr.shape) == 3:
        img_arr = img_arr[:,:,0]
    return img_arr

def get_hydra_nd_results(url):
    hydra_result = hydra_tg_falcon_client.hydra_tg(url)
#map hydra result to something equivalent to paperdoll output , return

    nd_result = neurodoll_falcon_client.nd(url,category_index=None,get_combined_results=True,multilabel_results=hydra_result)
    #todo - allow nd to accept multilabel results as input
    #convert multilabel results to ultimate_21 classes
    # nd_result = neurodoll_falcon_client.nd(url,category_index=None,get_multilabel_results=None,
    #                                        get_combined_results=None,get_layer_output=None,
    #                                        get_all_graylevels=None,threshold=None)

    # nd_result = neurodoll.combine_neurodoll_and_multilabel(url_or_np_array,multilabel_threshold=0.7,median_factor=1.0,
    #                                  multilabel_to_ultimate21_conversion=constants.binary_classifier_categories_to_ultimate_21,
    #                                  multilabel_labels=constants.binary_classifier_categories, face=None,
    #                                  output_layer = 'pixlevel_sigmoid_output',required_image_size=(224,224),
    #                                  do_graylevel_zeroing=True)

def image_to_name(url_or_filename_or_img_arr):
    if isinstance(url_or_filename_or_img_arr,basestring):
        name = basestring.replace('https://','').replace('http://','').replace('/','_')
    elif isinstance(url_or_filename_or_img_arr,np.ndarray):
        name = hash(str(url_or_filename_or_img_arr))
    print('name:'+name)
    return name

def get_groundtruth_for_tamaraberg_multilabel(labelfile='/data/jeremy/image_dbs/labels/labelfiles_tb/tb_cats_from_webtool_round2_train.txt',
                                              label_cats=constants.web_tool_categories_v2):
    with open(labelfile,'r') as fp:
        lines = fp.readlines()
    imgs_and_labels = [(line.split()[0],[int(i) for i in line.split()[1:]]) for line in lines]
    print(str(len(imgs_and_labels))+' images described in file '+labelfile)
    print('data looks like '+str(imgs_and_labels[0]))
    return imgs_and_labels

def do_imagelevel_comparison():
    imgs_and_labels = get_groundtruth_for_tamaraberg_multilabel()
    for image_file,label in imgs_and_labels:
        pd_result = get_pd_results(image_file)
        hydra_nd_result = get_hydra_nd_results(image_file)


def dl_images(source_domain='stylebook.de',text_filter='',dl_dir='/data/jeremy/image_dbs/golden/',in_docker=True,visual_output=False):
    '''
    dl everything in the images db, on the assumption that these are the  most relevant to test our answers to
    :return:
    '''

    if in_docker:
        db = pymongo.MongoClient('localhost',port=27017).mydb
    else:
        db = constants.db

    all = db.images.find({'domain':source_domain})
    for doc in all:
        url=doc['image_urls'][0]
        if text_filter in url[0]:
            print url
            Utils.get_cv2_img_array(url,convert_url_to_local_filename=True,download=True,download_directory=dl_dir)
        else:
            print('skipping '+url)

    #move the images with more than one person
    imutils.do_for_all_files_in_dir(imutils.one_person_per_image,dl_dir,visual_output=False)


def fast_hist(a, b, n):
    '''
    a confidence matrix - rows are guesses, cols are gt. n*a+b turns values into locations in confmat from 0...n**2-1,
    then histogram and reshape to get confmat
    :param a: flattened mask
    :param b: flattened ground truth mask
    :param n: number of categories
    :return:confmat
    '''
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)


def results_from_hist(hist,save_file='./summary_output.txt',info_string='',labels=constants.ultimate_21):
    # mean loss
    overall_acc = np.diag(hist).sum() / hist.sum()
    print '>>>', 'overall accuracy', overall_acc
    # per-class accuracy
    acc = np.diag(hist) / hist.sum(1)
    print '>>>',  'acc per class', str(acc)
    print '>>>',  'mean accuracy', np.nanmean(acc)
    # per-class IU
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print '>>>', 'mean IU', np.nanmean(iu)
    freq = hist.sum(1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    print '>>>', 'fwavacc', fwavacc
    mean_acc = np.nanmean(acc)
    mean_iou = np.nanmean(iu)
    results_dict = {'class_accuracy':acc.tolist(),'overall_acc':overall_acc.tolist(),'mean_acc':mean_acc.tolist(),'class_iou':iu.tolist(),'mean_iou':mean_iou.tolist(),'fwavacc':fwavacc.tolist()}
    if save_file:
        with open(save_file,'a+') as f:  #a+ creates if it doesnt exist
            f.write('net output '+' ' + info_string+ '\n')
            f.write('<br>\n')
            f.write('classes: \n')
            for i in range(len(labels)):
                f.write(str(i)+':'+labels[i]+' ')
            f.write('<br>\n')
            f.write('acc per class:'+ str(acc)+'\n')
            f.write('<br>\n')
            f.write('overall acc:'+ str(overall_acc)+'\n')
            f.write('<br>\n')
            f.write('mean acc:'+ str(np.nanmean(acc))+'\n')
            f.write('<br>\n')
            f.write('IU per class:'+ str(iu)+'\n')
            f.write('<br>\n')
            f.write('mean IU:'+ str(np.nanmean(iu))+'\n')
            f.write('<br>\n')
            f.write('fwavacc:'+ str((freq[freq > 0] * iu[freq > 0]).sum())+'\n')
            f.write('<br>\n')
            f.write('<br>\n')
    return results_dict


def do_seg_tests(net, iter, save_dir, n_images, output_layer='score', gt_layer='label',outfilename='net_output.txt',save_output=False,savedir='testoutput',labels=constants.pixlevel_categories_v3):
    n_cl = net.blobs[output_layer].channels
    hist, loss = compute_hist(n_images, output_layer, gt_layer,labels=labels)
    # mean loss
    print '>>>', 'Iteration', iter, 'loss', loss
    # overall accuracy
    overall_acc = np.diag(hist).sum() / hist.sum()
    print '>>>', 'Iteration', iter, 'overall accuracy', overall_acc
    # per-class accuracy
    acc = np.diag(hist) / hist.sum(1)
    print '>>>','Iteration', iter, 'acc per class', str(acc)
    print '>>>', 'Iteration', iter, 'mean accuracy', np.nanmean(acc)
    # per-class IU
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print '>>>','Iteration', iter, 'mean IU', np.nanmean(iu)
    freq = hist.sum(1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    print '>>>', 'Iteration', iter, 'fwavacc', \
            fwavacc
    mean_acc = np.nanmean(acc)
    mean_iou = np.nanmean(iu)
    results_dict = {'iter':iter,'loss':loss,'class_accuracy':acc.tolist(),'overall_acc':overall_acc.tolist(),'mean_acc':mean_acc.tolist(),'class_iou':iu.tolist(),'mean_iou':mean_iou.tolist(),'fwavacc':fwavacc.tolist()}
    jsonfile = outfilename[:-4]+'.json'
#    with open(jsonfile, 'a+') as outfile:
#        json.dump(results_dict, outfile)
#        outfile.close()

    with open(outfilename,'a') as f:
        f.write('>>>'+ str(datetime.now())+' Iteration:'+ str(iter)+ ' loss:'+ str(loss)+'\n')
        f.write('<br>\n')
        f.write('acc per class:'+ str(acc)+'\n')
        f.write('<br>\n')
        f.write('overall acc:'+ str(overall_acc)+'\n')
        f.write('<br>\n')
        f.write('mean acc:'+ str(np.nanmean(acc))+'\n')
        f.write('<br>\n')
        f.write('IU per class:'+ str(iu)+'\n')
        f.write('<br>\n')
        f.write('mean IU:'+ str(np.nanmean(iu))+'\n')
        f.write('<br>\n')
        f.write('fwavacc:'+ str((freq[freq > 0] * iu[freq > 0]).sum())+'\n')
        f.write('<br>\n')
        f.write('<br>\n')
    return results_dict

def results_to_html(outfilename,results_dict):
    acc = results_dict['class_accuracy']
    overall_acc = results_dict['overall_acc']
    mean_acc = results_dict['mean_acc']
    class_iou = results_dict['class_iou']
    mean_iou = results_dict['mean_iou']
    fwavacc = results_dict['fwavacc']
    with open(outfilename,'a') as f:
        f.write('<br>\n')
        f.write('acc per class:'+ str(acc)+'\n')
        f.write('<br>\n')
        f.write('overall acc:'+ str(overall_acc)+'\n')
        f.write('<br>\n')
        f.write('mean acc:'+ str(mean_acc)+'\n')
        f.write('<br>\n')
        f.write('IU per class:'+ str(class_iou)+'\n')
        f.write('<br>\n')
        f.write('mean IU:'+ str(mean_iou)+'\n')
        f.write('<br>\n')
        f.write('fwavacc:'+ str(fwavacc)+'\n')
        f.write('<br>\n')
        f.write('<br>\n')



if __name__ == "__main__":
#    Utils.map_function_on_dir(get_live_pd_results,'/data/jeremy/image_dbs/tg/pixlevel/pixlevel_fullsize_test/')
     all_pd_results()