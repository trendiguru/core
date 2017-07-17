__author__ = 'jeremy'

import msgpack
import requests
import cv2
import logging
logging.basicConfig(level=logging.INFO)



from trendi import constants
from trendi import Utils
from trendi.utils import imutils

def compare_bb_dicts(gt_dict,guess_dict,dict_format={'data':'data','bbox':'bbox','object':'object','confidence':'confidence'},iou_threshold=0.2):
    '''
    given 2 dicts of bbs - find bb in dict2 having most overlap for each bb in dict1 (assuming thats the gt)
    for each gt:
    find overlapping guesses where ovrerlap iou > 0.5
    take most confident guess
    any other overlaps are counted as false positive
    If there's no overlapping bb or the cat. is wrong thats a false negative.
    iou counts for average even if category is wrong (check this against standards...!)
    iou=0 if no guess box overlaps a gt box.
    extra detections with no iou>0.5 count as false pos, and contribute iou=0 to average
    mAP computed by averaging precision values on p-r curve at steps [0,0.1,...,1]
     To be more precise, we consider a slightly corrected PR curve, where for each curve point (p, r), if there is a
     different curve point (p', r') such that p' > p and r' >= r,
     we replace p with maximum p' of those points.
    there are no true negatives here to speak of

    see https://stackoverflow.com/questions/36274638/map-metric-in-object-detection-and-computer-vision
    and http://homepages.inf.ed.ac.uk/ckiw/postscript/ijcv_voc09.pdf

    Detections output by a method were assigned to ground
    truth objects satisfying the overlap criterion in order ranked by the (decreasing) confidence output. Multiple detections
    of the same object in an image were considered false detections e.g. 5 detections of a single object counted as 1 correct
    detection and 4 false detections  it was the responsibility of
    the participants system to filter multiple detections from its output


    What is still unclear to me is what is done with those GT boxes that are never detected (even if the confidence is 0)
    This means that there are certain recall values that the precision-recall curve will never reach, and this makes the
    average precision computation above undefined.

    I take GT's not detected as false negative

    That is not the only thing that is unclear. Consider a case where there are two predicted boxes (P1, P2) and two
    ground-truth boxes (T1, T2),
    where P2 has higher confidence than P1. Both P1 and P2 overlap T1. Since P2 has the higher confidence, it is clear
    that P2 should be considered
    the match for T1. What is not given is if P1 also has some IOU overlap with T2, but lower than the IOU with T1, should
    P1 be given a second chance to try to match itself to T2,
    or should it not

    I am allowing this in the code below. only the matching (highest conf) prediction is marked as 'already matched'

    Another case:
    Say that P2 also overlaps T2 and has higher IOU than with T1, then it seems that P2 should be matched with T2 and not T1
    this isn't taken care of in code below...

    :param dict1:ground truth in 'api form' {'data': [{ 'object': 'bag', 'bbox': [454, 306, 512, 360]},...,]}
                guess in same form but with confidence (gt can also have confidence 1.0)
            bbox here is xywh , aka x1 y1 w h , coords are 'regular' image coords
            (origin is top left, positive x goes right and pos y goes down)
    :param dict2:guess in 'api form'
    :param dict_format - this lets you use dicts in different formats, just substitute whatver term is used into the
    dict

    e.g.         if the dict uses 'x_y_w_h' instead of 'bbox' and 'annotations' instead of 'data' and 'label'
     instead
     of 'object'
    then dict_format = {'data':'annotations', 'bbox':'w_y_w_h','object:'label'}
    :return:  n_true_positive, n_false_neg, n_false_pos, avg_iou
    '''


    gt_data=gt_dict[dict_format['data']]
    guess_data=guess_dict[dict_format['data']]
    true_pos = 0
    false_pos = 0 #aka lonely hypothesis
    false_neg = 0  #aka lonely ground truth
    #there are no true negatives here to speak of
    tot_gt_objects = 0
    iou_tot = 0
    n_detections = 0
    detections_over_threshold = 0
    obj_kw = dict_format['object']
    bb_kw = dict_format['bbox']
    conf_kw = dict_format['confidence']
    for gt_detection in gt_data:
        most_confident_detection = None
        highest_confidence = 0
        iou_of_most_confident_detection = 0

        correct_object_guesses = [guess for guess in guess_data if guess[obj_kw]==gt_detection[obj_kw]]
        print('matching items for {}:{} '.format(gt_detection,correct_object_guesses))
        for guess_detection in correct_object_guesses:
            if 'already_matched' in guess_data:
                print('already matched guess {}'.format(guess_detection))
                continue
            iou = Utils.intersectionOverUnion(gt_detection[bb_kw],guess_detection[bb_kw])
            print('checking gt {} {} vs {} {} conf {}, iou {}'.format(gt_detection[bb_kw],
                                                            gt_detection[obj_kw],
                                                            guess_detection[bb_kw],
                                                            guess_detection[obj_kw],guess_detection[conf_kw],iou))
#            if iou>best_iou :
            if guess_detection[conf_kw]>highest_confidence and iou>0:
                highest_confidence = guess_detection[conf_kw]
                most_confident_detection = guess_detection
                iou_of_most_confident_detection = iou
                print('most confident so far')
        if most_confident_detection is not None:
            n_detections += 1
            most_confident_detection['already_matched']=True #this gets put into original guess_detection
            gt_detection['already_matched']=True #this gets put into original gt_detection
            if iou_of_most_confident_detection > iou_threshold:
                detections_over_threshold += 1
                true_pos += 1
            else:
                false_neg += 1  # best guess has iou < 0.5
                print('best guess has iou {} < threshold {}'.format(iou_of_most_confident_detection,iou_threshold))
        else:
            false_neg += 1  #completely unmatched ground truth
            print('no overlapping box found')
        tot_gt_objects += 1
        iou_tot += iou_of_most_confident_detection
        print('tp {} fn {} gt objects seen {} avg_iou {} tot_iou {}'.format(true_pos,false_neg,tot_gt_objects,iou_tot/tot_gt_objects,iou_tot))
    #check for extra guess detections
    for guess_detection in guess_data:
        if not 'already_matched' in guess_detection:
            print('{} is a false pos'.format(guess_detection))
            false_pos += 1
    for gt_detection in gt_data:
        if not 'already_matched' in gt_detection:
            print('{} is a false neg'.format(gt_detection))
            false_neg += 1
    iou_avg = iou_tot/n_detections
    print('final tp {} fp {} fn {} gt objects seen {} avg_iou {}'.format(true_pos,false_pos,false_neg,tot_gt_objects,iou_avg))
    return {'tp':true_pos,'tn':false_pos,'fn':false_neg,'iou_avg':iou_avg}

def test_compare_bb_dicts():
    img = '/home/jeremy/projects/core/images/2017-07-06_09-15-41-308.jpeg'
    gt = {   "data" : [
    { "object" : "Van",
      "bbox" : [1428,466, 98, 113 ]     },
    { "object" : "mazda",
      "bbox" : [1306, 485, 83,64 ]     },
    { "object" : "vw",
      "bbox" : [1095,453,103,68 ]     },
    { "object" : "austin",
      "bbox" : [1204, 479, 96, 59 ]     },
    { "object" : "mercedes",
      "bbox" : [1010, 468, 79, 42 ]     },
    { "object" : "subaru",
      "bbox" : [760, 864,586,158 ] }  ] }

    guess =  {   "data" : [
    { "object" : "Van",
      "bbox" : [1400,500, 70, 70 ],'confidence':0.9     },
    { "object" : "Van",
      "bbox" : [1440,490, 80, 90 ],'confidence':0.8     },
    { "object" : "mazda",
      "bbox" : [1300, 385, 40,50 ] ,'confidence':0.8    },
    { "object" : "XX",
      "bbox" : [1000,433,103,68 ] ,'confidence':0.9    },
    { "object" : "austin",
      "bbox" : [1200, 450, 100, 100 ] ,'confidence':0.8     },
    { "object" : "vw",
      "bbox" : [1100, 490, 30, 50 ]  ,'confidence':0.8    },
    { "object" : "ferrari",
      "bbox" : [1060, 350, 30, 60 ] ,'confidence':0.8     },
    { "object" : "subaru",
      "bbox" : [750, 869,586,158 ],'confidence':0.7  },
        { "object" : "subaru",
      "bbox" : [740, 840,570,140 ],'confidence':0.8  }  ] }

    img_arr = cv2.imread(img)
    if img_arr is None:
        print('got none for '+img)
    for gt_obj in gt['data']:
        img_arr = imutils.bb_with_text(img_arr,gt_obj['bbox'],gt_obj['object'],boxcolor=[255,0,0])
    for obj in guess['data']:
        img_arr = imutils.bb_with_text(img_arr,obj['bbox'],obj['object']+' '+str(obj['confidence']),boxcolor=[0,255,0])
    cv2.imshow('img',img_arr)
    cv2.waitKey(0)

    compare_bb_dicts(gt,guess)

def get_classes(detection_dicts,object_keyword='object'):
    class_list=[]
    for detection in detection_dicts:
        if not object_keyword in detection:
            logging.warning('did not find object kw {} in detection {}'.format(object_keyword,detection))
            continue

def mAP_and_iou(gt_detections,guess_detections,dict_format={'data':'data','bbox':'bbox','object':'object','confidence':'confidence'}):
    gt_classes = get_classes(gt_detections,dict_format['object'])
    guess_classes = get_classes(guess_detections,dict_format['object'])

def precision_accuracy_recall(caffemodel,solverproto,outlayer='label',n_tests=100):
    #TODO dont use solver to get inferences , no need for solver for that
    caffe.set_mode_gpu()
    caffe.set_device(1)

    workdir = './'
    snapshot = 'snapshot'
#    caffemodel = '/home/jeremy/caffenets/multilabel/vgg_ilsvrc_16_multilabel_2/snapshot/train_iter_240000.caffemodel'
#    caffemodel = '/home/jeremy/caffenets/multilabel/vgg_ilsvrc_16_multilabel_2/snapshot/train_iter_340000.caffemodel'
    model_base = caffemodel.split('/')[-1]
    p_all = []
    r_all = []
    a_all = []
    n_all = []
#    for t in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.92,0.95,0.98]:
    thresh = [0.1,0.5,0.6,0.7,0.8,0.9,0.95]
#    thresh = [0.1,0.5,0.95]
    protoname = solverproto.replace('.prototxt','')
    netname = get_netname(solverproto)
    if netname:
        dir = 'multilabel_results-'+netname+'_'+model_base.replace('.caffemodel','')
        dir = dir.replace('"','')  #remove quotes
        dir = dir.replace(' ','')  #remove spaces
        dir = dir.replace('\n','')  #remove newline
        dir = dir.replace('\r','')  #remove return
    else:
        dir = 'multilabel_results-'+protoname+'_'+model_base.replace('.caffemodel','')
        dir = dir.replace('"','')  #remove quotes
        dir = dir.replace(' ','')  #remove spaces
        dir = dir.replace('\n','')  #remove newline
        dir = dir.replace('\r','')  #remove return

    print('dir to save stuff in : '+str(dir))
    Utils.ensure_dir(dir)
#    open_html(model_base,dir=dir)
    positives = True
    for t in thresh:
        p,r,a,tp,tn,fp,fn = check_accuracy(solverproto, caffemodel, threshold=t, num_batches=n_tests,outlayer=outlayer)
        p_all.append(p)
        r_all.append(r)
        a_all.append(a)
        n_occurences = [tp[i]+fn[i] for i in range(len(tp))]
        n_all.append(n_occurences)
        write_textfile(p,r,a,tp,tn,fp,fn,t,model_base,dir=dir)
        write_html(p,r,a,n_occurences,t,model_base,positives=positives,dir=dir,tp=tp,tn=tn,fp=fp,fn=fn)
        positives = False
    close_html(model_base,dir=dir)

    p_all_np = np.transpose(np.array(p_all))
    r_all_np = np.transpose(np.array(r_all))
    a_all_np = np.transpose(np.array(a_all))
    labels = constants.web_tool_categories
    plabels = [label + 'precision' for label in labels]
    rlabels = [label + 'recall' for label in labels]
    alabels = [label + 'accuracy' for label in labels]

    important_indices = [3,5,7,10,11,13,17]
    #cardigan  dress footwear jeans pants skirt top
    #['bag', 'belt', 'blazer','cardigan','coat','dress', 'eyewear', 'footwear', 'hat','jacket',
     #                  'jeans','pants','shorts', 'skirt','stocking','suit','sweater','top','scarf','womens_swimwear_bikini',
      #                 'womens_swimwear_nonbikini']

    p_important = [p_all_np[i] for i in important_indices]
    r_important = [r_all_np[i] for i in important_indices]
    a_important = [a_all_np[i] for i in important_indices]
    labels_important = [labels[i] for i in important_indices]
    for i in range(len(important_indices)):
        print(constants.web_tool_categories[i]+' p:'+str(p_important[i])+' r:'+str(r_important[i])+' a:'+str(a_important[i]) )
    thresh_all_np = np.array(thresh)
    print('shape:'+str(p_all_np.shape))
    print('len:'+str(len(p_important)))

    markers = [ '^','<','v','^','8','o',   '.','x','|',
                          '+', 0, '4', 3,4, 'H', '3', 'p', 'h', '*', 7,'', 5, ',', '2', 1, 6, 's', 'd', '1','_',  2,' ', 'D']
    markers = ['.','x','|', '^',
                '+','<',
                0,'v',
               '4', 3,'^',
                '8',
                4,'o',
                'H', '3', 'p',  '*','h',
               7,'', 5, ',', '2', 1, 6, 's', 'd', '1','_',  2,' ', 'D']
    markers_important = ['^','<','v','^', '8','o','H', '3', 'p',  '*','h']


    for i in range(len(p_important)):
        plt.subplot(311)
        print('plotting {} vs {}'.format(p_all_np[i,:],thresh_all_np))
        plt.plot(thresh_all_np,p_important[i],label=labels_important[i],linestyle='None',marker=markers_important[i])
        plt.subplot(312)   #
        plt.plot(thresh_all_np,r_important[i],label=labels_important[i],linestyle='None',marker=markers_important[i])
        plt.subplot(313)
        plt.plot(thresh_all_np,a_important[i],label=labels_important[i],linestyle='None',marker=markers_important[i])
#        plt.plot(thresh_all_np,a_all_np[i],label=labels_important[i],linestyle='None',marker=markers_important[i])
#        plt.plot(thresh_all_np,p_all_np[i,:],label=labels[i],marker=markers[i])
#        plt.subplot(312)   #
#        plt.plot(thresh_all_np,r_all_np[i,:],label=labels[i],linestyle='None',marker=markers[i])
#        plt.subplot(313)
#        plt.plot(thresh_all_np,a_all_np[i,:],label=labels[i],linestyle='None',marker=markers[i])
    plt.subplot(311)
    plt.title('results '+model_base)
    plt.xlabel('threshold')
    plt.ylabel('precision')
    plt.grid(True)
    plt.ylim((0,1))
    plt.subplot(312)   #
    plt.xlabel('threshold')
    plt.ylabel('recall')
    plt.grid(True)
    plt.ylim((0,1))
    plt.subplot(313)
    plt.xlabel('threshold')
    plt.ylabel('accuracy')
    plt.ylim((0,1))
    plt.grid(True)

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.1))
    plt.show()#

    figname = os.path.join(dir,model_base+'.png')
    print('saving figure:'+str(figname))
    plt.savefig(figname, bbox_inches='tight')
#
    summary_html(dir)
  #  print 'Baseline accuracy:{0:.4f}'.format(check_baseline_accuracy(solver.test_nets[0], 10,batch_size = 20))

def bb_output_using_gunicorn(url_or_np_array):
    print('starting get_multilabel_output_using_nfc')
    multilabel_dict = nfc.pd(url_or_np_array, get_multilabel_results=True)
    logging.debug('get_multi_output:dict from falcon dict:'+str(multilabel_dict))
    if not multilabel_dict['success']:
        logging.warning('did not get nfc pd result succesfully')
        return
    multilabel_output = multilabel_dict['multilabel_output']
    logging.debug('multilabel output:'+str(multilabel_output))
    return multilabel_output #

def bb_output_yolo_using_api(url_or_np_array,CLASSIFIER_ADDRESS=constants.YOLO_HLS_CLASSIFIER_ADDRESS,roi=None):
    print('starting bb_output_api at addr '+str(CLASSIFIER_ADDRESS))
#    CLASSIFIER_ADDRESS =   # "http://13.82.136.127:8082/hls"
    print('using yolo api addr '+str(CLASSIFIER_ADDRESS))
    if isinstance(url_or_np_array,basestring): #got a url (or filename, but not dealing with that case)
        data = {"imageUrl": url_or_np_array}
        print('using imageUrl as data')
    else:
        img_arr = Utils.get_cv2_img_array(url_or_np_array)
        data = {"image": img_arr} #this was hitting 'cant serialize' error
        print('using imgage as data')
    if roi:
        print "Make sure roi is a list in this order [x1, y1, x2, y2]"
        data["roi"] = roi
    serialized_data = msgpack.dumps(data)
#    resp = requests.post(CLASSIFIER_ADDRESS, data=serialized_data)
    result = requests.get(CLASSIFIER_ADDRESS,params=data)
    if result.status_code is not 200:
       print("Code is not 200")
#     else:
#         for chunk in result.iter_content():
#             print(chunk)
# #            joke = requests.get(JOKE_URL).json()["value"]["joke"]

#    resp = requests.post(CLASSIFIER_ADDRESS, data=data)
    c=  result.content
    #content should be roughly in form
#    {"data":
    # [{"confidence": 0.366, "object": "car", "bbox": [394, 49, 486, 82]},
    # {"confidence": 0.2606, "object": "car", "bbox": [0, 116, 571, 462]}, ... ]}
    if not 'data' in c:
        print('didnt get data in result from {} on sendng {}'.format(CLASSIFIER_ADDRESS,data))
    return data
    # t = result.text
    # print('content {} text {}'.format(c,t))

if __name__ == " __main__":
    print('main')
    test_compare_bb_dicts()