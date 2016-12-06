__author__ = 'jeremy'

from trendi.paperdoll import neurodoll_falcon_client as nfc
from trendi import constants
from trendi import Utils

def combine_pixlevel_and_multilabel(url_or_np_array,multilabel=None,pixel_graylevels=None,multilabel_threshold=0.7,median_factor=1.0,
                                     multilabel_to_ultimate21_conversion=constants.binary_classifier_categories_to_ultimate_21,
                                     multilabel_labels=constants.binary_classifier_categories, face=None,
                                     output_layer = 'pixlevel_sigmoid_output',required_image_size=(224,224),
                                     do_graylevel_zeroing=True):
    if pixel_graylevels is None or multilabel is None:
        multilabel_dict = nfc.pd(url_or_np_array, get_multilabel_results=True,get_all_graylevels=True)
        print('get_multi_output:dict from falcon dict:'+str(multilabel_dict))
        if not multilabel_dict['success']:
            print('did not get nfc pd result succesfully')
            return
        if pixel_graylevels is None:
            pixel_graylevels = multilabel_dict['all_graylevel_output']
        if multilabel is None:
            multilabel = multilabel_dict['multilabel_output']

    retval = combine_pixlevel_and_multilabel_using_graylevel(pixel_graylevels,multilabel,multilabel_threshold=multilabel_threshold,median_factor=median_factor,
                                     multilabel_to_ultimate21_conversion=multilabel_to_ultimate21_conversion,
                                     multilabel_labels=multilabel_labels, face=face,
                                     output_layer = output_layer,required_image_size=required_image_size,
                                     do_graylevel_zeroing=do_graylevel_zeroing)
    return retval


def combine_pixlevel_and_multilabel_using_graylevel(pixel_graylevels,multilabel,multilabel_threshold=0.7,median_factor=1.0,
                                     multilabel_to_ultimate21_conversion=constants.binary_classifier_categories_to_ultimate_21,
                                     multilabel_labels=constants.binary_classifier_categories, face=None,
                                     output_layer = 'pixlevel_sigmoid_output',required_image_size=(224,224),
                                     do_graylevel_zeroing=True):
    print('combining multilabel w. neurodoll using nfc watch out, required imsize:'+str(required_image_size))
    thedir = './images'
    Utils.ensure_dir(thedir)
    if isinstance(url_or_np_array, basestring):
        image = url_to_image(url_or_np_array)
        orig_filename = os.path.join(thedir,url_or_np_array.split('/')[-1]).replace('.jpg','')
    elif type(url_or_np_array) == np.ndarray:
        hash = hashlib.sha1()
        hash.update(str(time.time()))
        name_base = 'orig'+hash.hexdigest()[:10]
        orig_filename = os.path.join(thedir,name_base)
        image = url_or_np_array
    if image is None:
        logging.debug('got None in grabcut_using_neurodoll_output')
    print('writing orig to '+orig_filename+'.jpg')
    cv2.imwrite(orig_filename+'.jpg',image)
    multilabel = get_multilabel_output(url_or_np_array)
#    multilabel = get_multilabel_output_using_nfc(url_or_np_array)
    #take only labels above a threshold on the multilabel result
    #possible other way to do this: multiply the neurodoll mask by the multilabel result and threshold that product
    if multilabel is None:
        logging.debug('None result from multilabel')
        return None
    thresholded_multilabel = [ml>multilabel_threshold for ml in multilabel] #
    logging.info('orig label:'+str(multilabel)+' len:'+str(len(multilabel)))
#    print('incoming label:'+str(multilabel))
#    logging.info('thresholded label:'+str(thresholded_multilabel))
    for i in range(len(thresholded_multilabel)):
        if thresholded_multilabel[i]:
            logging.info(multilabel_labels[i]+' is over threshold')
#    print('multilabel to u21 conversion:'+str(multilabel_to_ultimate21_conversion))
#    print('multilabel labels:'+str(multilabel_labels))

    #todo - this may be wrong later if we start taking both nd and multilabel into acct. Maybe ml thinks theres nothing there but nd thinks there is...
    if np.equal(thresholded_multilabel,0).all():  #all labels 0 - nothing found
        logging.debug('no items found')
        return #


    pixlevel_categorical_output = graylevel_nd_output.argmax(axis=2) #the returned mask is HxWxC so take max along C
    pixlevel_categorical_output = threshold_pixlevel(pixlevel_categorical_output) #threshold out the small areas
    print('before graylevel zeroing:')
    count_values(pixlevel_categorical_output,labels=constants.ultimate_21)

    if do_graylevel_zeroing:
        graylevel_nd_output = zero_graylevels_not_in_ml(graylevel_nd_output,multilabel,threshold=0.7)

    pixlevel_categorical_output = graylevel_nd_output.argmax(axis=2) #the returned mask is HxWxC so take max along C
    pixlevel_categorical_output = threshold_pixlevel(pixlevel_categorical_output) #threshold out the small areas
    print('after graylevel zeroing:')
    count_values(pixlevel_categorical_output,labels=constants.ultimate_21)
    foreground = np.array((pixlevel_categorical_output>0) * 1)
    background = np.array((pixlevel_categorical_output==0) * 1)
    #    item_masks =  nfc.pd(image, get_all_graylevels=True)
    logging.debug('shape of pixlevel categorical output:'+str(pixlevel_categorical_output.shape))
    logging.debug('n_fg {} n_bg {} tot {} w*h {}'.format(np.sum(foreground),np.sum(background),np.sum(foreground)+np.sum(background),pixlevel_categorical_output.shape[0]*pixlevel_categorical_output.shape[1]))

    first_time_thru = True  #hack to dtermine image size coming back from neurodoll

 #   final_mask = np.zeros([224,224])
    final_mask = np.zeros(pixlevel_categorical_output.shape[:])
    print('final_mask shape '+str(final_mask.shape))

    if face:
        y_split = face[1] + 3 * face[3]
    else:
        # BETTER TO SEND A FACE
        y_split = np.round(0.4 * final_mask.shape[0])
    print('y split {} face {}'.format(y_split,face))

    #the grabcut results dont seem too hot so i am moving to a 'nadav style' from-nd-and-ml-to-results system
    #namely : for top , decide if its a top or dress or jacket
    # for bottom, decide if dress/pants/skirt
    #decide on one bottom
 #   for i in range(len(thresholded_multilabel)):
 #       if multilabel_labels[i] in ['dress', 'jeans','shorts','pants','skirt','suit','overalls'] #missing from list is various swimwear which arent getting returned from nd now anyway

#############################################################################################
#Make some conclusions nadav style.
#Currently the decisions are based only on ml results without taking into acct the nd results.
#In future possibly inorporate nd as well, first do a head-to-head test of nd vs ml
#############################################################################################
    #1. take winning upper cover,  donate losers to winner
    #2. take winning upper under, donate losers to winner
    #3. take winning lower cover, donate losers to winner.
    #4. take winning lower under, donate losers to winner
    #5. decide on whole body item (dress, suit, overall) vs. non-whole body (two part e.g. skirt+top) items.
    #6. if wholebody beats two-part - donate all non-whole-body pixels to whole body (except upper-cover (jacket/blazer etc)  and lower under-stockings)
    #?  if no upper cover and no upper under and no whole-body: take max of all those and donate losers to winner

    #upper_cover: jacket, coat, blazer etc
    #upper under: shirt, top, blouse etc
    #lower cover: skirt, pants, shorts
    #lower under: tights, leggings

    whole_body_indexlist = [multilabel_labels.index(s) for s in  ['dress', 'suit','overalls']] #swimsuits could be added here
    upper_cover_indexlist = [multilabel_labels.index(s) for s in  ['cardigan', 'coat','jacket','sweater','sweatshirt']]
    upper_under_indexlist = [multilabel_labels.index(s) for s in  ['top']]
    lower_cover_indexlist = [multilabel_labels.index(s) for s in  ['jeans','pants','shorts','skirt']]
    lower_under_indexlist = [multilabel_labels.index(s) for s in  ['stocking']]

    final_mask = np.copy(pixlevel_categorical_output)
    logging.info('size of final mask '+str(final_mask.shape))

    print('wholebody indices:'+str(whole_body_indexlist))
    for i in whole_body_indexlist:
        print multilabel_labels[i]
    whole_body_ml_values = np.array([multilabel[i] for i in whole_body_indexlist])
    print('wholebody ml_values:'+str(whole_body_ml_values))
    whole_body_winner = whole_body_ml_values.argmax()
    whole_body_winner_value=whole_body_ml_values[whole_body_winner]
    whole_body_winner_index=whole_body_indexlist[whole_body_winner]
    print('winning index:'+str(whole_body_winner)+' mlindex:'+str(whole_body_winner_index)+' value:'+str(whole_body_winner_value))

    print('uppercover indices:'+str(upper_cover_indexlist))
    for i in upper_cover_indexlist:
        print multilabel_labels[i]
    upper_cover_ml_values = np.array([multilabel[i] for i in  upper_cover_indexlist])
    print('upper_cover ml_values:'+str(upper_cover_ml_values))
    upper_cover_winner = upper_cover_ml_values.argmax()
    upper_cover_winner_value=upper_cover_ml_values[upper_cover_winner]
    upper_cover_winner_index=upper_cover_indexlist[upper_cover_winner]
    print('winning upper_cover:'+str(upper_cover_winner)+' mlindex:'+str(upper_cover_winner_index)+' value:'+str(upper_cover_winner_value))

    print('upperunder indices:'+str(upper_under_indexlist))
    for i in upper_under_indexlist:
        print multilabel_labels[i]
    upper_under_ml_values = np.array([multilabel[i] for i in  upper_under_indexlist])
    print('upper_under ml_values:'+str(upper_under_ml_values))
    upper_under_winner = upper_under_ml_values.argmax()
    upper_under_winner_value=upper_under_ml_values[upper_under_winner]
    upper_under_winner_index=upper_under_indexlist[upper_under_winner]
    print('winning upper_under:'+str(upper_under_winner)+' mlindex:'+str(upper_under_winner_index)+' value:'+str(upper_under_winner_value))

    print('lowercover indices:'+str(lower_cover_indexlist))
    for i in lower_cover_indexlist:
        print multilabel_labels[i]
    lower_cover_ml_values = np.array([multilabel[i] for i in lower_cover_indexlist])
    print('lower_cover ml_values:'+str(lower_cover_ml_values))
    lower_cover_winner = lower_cover_ml_values.argmax()
    lower_cover_winner_value=lower_cover_ml_values[lower_cover_winner]
    lower_cover_winner_index=lower_cover_indexlist[lower_cover_winner]
    print('winning lower_cover:'+str(lower_cover_winner)+' mlindex:'+str(lower_cover_winner_index)+' value:'+str(lower_cover_winner_value))

    print('lowerunder indices:'+str(lower_under_indexlist))
    for i in lower_under_indexlist:
        print multilabel_labels[i]
    lower_under_ml_values = np.array([multilabel[i] for i in  lower_under_indexlist])
    print('lower_under ml_values:'+str(lower_under_ml_values))
    lower_under_winner = lower_under_ml_values.argmax()
    lower_under_winner_value=lower_under_ml_values[lower_under_winner]
    lower_under_winner_index=lower_under_indexlist[lower_under_winner]
    print('winning lower_under:'+str(lower_under_winner)+' mlindex:'+str(lower_under_winner_index)+' value:'+str(lower_under_winner_value))

    #for use later, decide on a winner between upper cover and upper under
    if upper_under_winner_value > upper_cover_winner_value:
        upper_winner_value = upper_under_winner_value
        upper_winner_index = upper_under_winner_index
    else:
        upper_winner_value = upper_cover_winner_value
        upper_winner_index = upper_cover_winner_index
    #for use later, decide on a winner between lower cover and lower under
    if lower_under_winner_value > lower_cover_winner_value:
        lower_winner_value = lower_under_winner_value
        lower_winner_index = lower_under_winner_index
    else:
        lower_winner_value = lower_cover_winner_value
        lower_winner_index = lower_cover_winner_index
    upper_winner_nd_index = multilabel_to_ultimate21_conversion[upper_winner_index]
    lower_winner_nd_index = multilabel_to_ultimate21_conversion[lower_winner_index]
    print('upper winner {} nd {} val {} lower winner {} nd {} val {}'.format(upper_winner_index,upper_winner_nd_index,upper_winner_value,
                                                                             lower_winner_index,lower_winner_nd_index,lower_winner_value))
#1. take max upper cover , donate losers to winner
#this actually might not be always right, e.g. jacket+ sweater
#todo  - #1 - 4 can be put into a function since they are nearly identical
    neurodoll_upper_cover_index = multilabel_to_ultimate21_conversion[upper_cover_winner_index]
    if neurodoll_upper_cover_index is None:
        logging.warning('nd upper cover index {}  has no conversion '.format(upper_cover_winner_index))
    else:
        n = len(final_mask[final_mask==neurodoll_upper_cover_index])
        logging.debug('donating to upper cover winner, initial n :'+str(n)+' for ndindex '+str(neurodoll_upper_cover_index)+' ml index '+str(upper_cover_winner_index)+ ', checking mls '+str(upper_cover_indexlist))
        for i in upper_cover_indexlist: #whole_body donated to upper_under
            nd_index = multilabel_to_ultimate21_conversion[i]
            if nd_index is None:
                logging.warning('ml index {} has no conversion '.format(i))
                continue
            x = final_mask[final_mask==nd_index]
            final_mask[final_mask==nd_index] = neurodoll_upper_cover_index
            n = len(final_mask[final_mask==neurodoll_upper_cover_index])
            logging.info('upper cover ndindex {} {} donated to upper cover winner nd {} , now {} pixels, lenx {} '.format(nd_index,constants.ultimate_21[nd_index],neurodoll_upper_cover_index, n,len(x)))

#2. take max upper under, donate losers to winner
    neurodoll_upper_under_index = multilabel_to_ultimate21_conversion[upper_under_winner_index]
    if neurodoll_upper_under_index is None:
        logging.warning('nd upper cover index {}  has no conversion '.format(upper_under_winner_index))
    else:
        n = len(final_mask[final_mask==neurodoll_upper_under_index])
        logging.debug('donating to upper under winner, initial n :'+str(n)+' for ndindex '+str(neurodoll_upper_under_index)+' ml index '+str(upper_under_winner_index)+ ', checking mls '+str(upper_under_indexlist))
        for i in upper_under_indexlist: #upper under losers donated to upper under winner
            nd_index = multilabel_to_ultimate21_conversion[i]
            print('nd index {} ml index {}'.format(nd_index,i))
            if nd_index is None:
                logging.warning('ml index {} has no conversion '.format(i))
                continue
            final_mask[final_mask==nd_index] = neurodoll_upper_under_index
            n = len(final_mask[final_mask==neurodoll_upper_under_index])
            logging.info('upper under ndindex {} {} donated to upper under winner nd {}, now {} pixels'.format(nd_index,constants.ultimate_21[nd_index],neurodoll_upper_under_index,n))

#3. take max lower cover, donate losers to winner.
    neurodoll_lower_cover_index = multilabel_to_ultimate21_conversion[lower_cover_winner_index]
    if neurodoll_lower_cover_index is None:
        logging.warning('nd lower cover index {}  has no conversion '.format(lower_cover_winner_index))
    else:
        n = len(final_mask[final_mask==neurodoll_lower_cover_index])
        logging.debug('donating to lower cover winner, initial n :'+str(n)+' for ndindex '+str(neurodoll_lower_cover_index)+' ml index '+str(lower_cover_winner_index)+ ', checking mls '+str(lower_cover_indexlist))
        for i in lower_cover_indexlist: #lower cover losers donated to lower cover winner
            nd_index = multilabel_to_ultimate21_conversion[i]
            if nd_index is None:
                logging.warning('ml index {} has no conversion '.format(i))
                continue
            final_mask[final_mask==nd_index] = neurodoll_lower_cover_index
            n = len(final_mask[final_mask==neurodoll_lower_cover_index])
            logging.info('lower cover ndindex {} {} donated to lower cover winner nd {}, now {} pixels'.format(nd_index,constants.ultimate_21[nd_index],neurodoll_lower_cover_index,n))

#4. take max lower under, donate losers to winner.
    neurodoll_lower_under_index = multilabel_to_ultimate21_conversion[lower_under_winner_index]
    if neurodoll_lower_under_index is None:
        logging.warning('nd lower under index {}  has no conversion '.format(lower_under_winner_index))
    else:
        n = len(final_mask[final_mask==neurodoll_lower_under_index])
        logging.debug('donating to lower under winner, initial n :'+str(n)+' for ndindex '+str(neurodoll_lower_under_index)+' ml index '+str(lower_under_winner_index)+ ', checking mls '+str(lower_under_indexlist))
        for i in lower_under_indexlist: #lower under losers donated to lower under winner
            nd_index = multilabel_to_ultimate21_conversion[i]
            if nd_index is None:
                logging.warning('ml index {} has no conversion '.format(i))
                continue
            final_mask[final_mask==nd_index] = neurodoll_lower_under_index
            n = len(final_mask[final_mask==neurodoll_lower_under_index])
            logging.info('lower under ndindex {} {} donated to lower under winner nd {}, now {} pixels'.format(nd_index,constants.ultimate_21[nd_index],neurodoll_lower_under_index,n))

    logging.debug('after step 4, pixelcounts look like:')
    count_values(final_mask,labels=constants.ultimate_21)

#########################
# 5. WHOLEBODY VS TWO-PART
# decide on whole body item (dress, suit, overall) vs. non-whole body items.
# case 1 wholebody>upper_under>lower_cover
# case 2 upper_under>wholebody>lower_cover
# case 3 lower_cover>wholebody>upper-under
# case 4 lower_cover,upper_under > wholebody
#consider reducing this to nadav's method:
#    whole_sum = np.sum([item.values()[0] for item in mask_sizes['whole_body']])
#    partly_sum = np.sum([item.values()[0] for item in mask_sizes['upper_under']]) +\
#                 np.sum([item.values()[0] for item in mask_sizes['lower_cover']])
#                if whole_sum > partly_sum:
#    donate partly to whole
# its a little different tho since in multilabel you cant compare directly two items to one , e.g. if part1 = 0.6, part2 = 0.6, and whole=0.99, you
# should prob go with whole even tho part1+part2>whole
#########################
    neurodoll_wholebody_index = multilabel_to_ultimate21_conversion[whole_body_winner_index]
    if neurodoll_wholebody_index is None:
        logging.warning('nd wholebody index {} ml index {} has no conversion '.format(neurodoll_wholebody_index,whole_body_winner_index))

#first case - wholebody > upper_under > lowercover
#donate all non-whole-body pixels to whole body (except upper-cover (jacket/blazer etc)  and lower under-stockings)
    elif (whole_body_winner_value>upper_under_winner_value) and (whole_body_winner_value>lower_cover_winner_value) and whole_body_winner_value>multilabel_threshold:
        logging.info('case 1. one part {} wins over upper cover {} and lower cover {}'.format(whole_body_winner_value,upper_cover_winner_value,lower_cover_winner_value))
        n = len(final_mask[final_mask==neurodoll_wholebody_index])
        logging.info('n in final mask from wholebody alone:'+str(n))
        for i in upper_cover_indexlist:
            #jackets etc can occur with dress/overall so dont donate these
            pass
        for i in upper_under_indexlist:  #donate upper_under to whole_body
    #todo fix the case of suit (which can have upper_under)
            #ideally, do this for dress - suit and overalls can have upper_under
            nd_index = multilabel_to_ultimate21_conversion[i]
            if nd_index is None:
                logging.debug('upper cover nd index for {} has no conversion '.format(i))
                continue
            #add upper cover item to wholebody mask
            final_mask[final_mask==nd_index] = neurodoll_wholebody_index
            logging.info('adding upperunder nd index {} '.format(nd_index))
            n = final_mask[final_mask==neurodoll_wholebody_index]
            logging.info('n in final mask from wholebody:'+str(n))
        for i in lower_cover_indexlist: #donate lower_cover to whole_body
            nd_index = multilabel_to_ultimate21_conversion[i]
            final_mask[final_mask==nd_index] = neurodoll_wholebody_index
            logging.info('adding lowercover nd index {} '.format(nd_index))
            n = final_mask[final_mask==neurodoll_wholebody_index]
            logging.info('n in final mask from wholebody alone:'+str(n))
        for i in lower_under_indexlist:
            #not doing this for stockings which is currently the only lower under
            pass
        logging.debug('after case one pixel values look like')
        count_values(final_mask,labels=constants.ultimate_21)


# second case - upper_under > wholebody > lowercover
# here its not clear who to sack - the wholebody or the upper_under
# so i arbitrarily decided to sack th whole_body in favor of the upper_under since upper_under is higher
# EXCEPT if the wholebody is overalls , in which case keep overalls, upper_under and upper_cover, donate  lower_cover/under to overalls
# otherwise  if wholebody is e.g. dress then add dress to upper_under and lower_cover

    elif (whole_body_winner_value<upper_under_winner_value) and (whole_body_winner_value>lower_cover_winner_value) and (whole_body_winner_value>multilabel_threshold):
        logging.info('case 2. one part {} < upper under {} but > lower cover {}'.format(whole_body_winner_value,upper_under_winner_value,lower_cover_winner_value))
#if overalls, donate loewr_cover and lower_under to overalls
        if whole_body_winner_index == multilabel_labels.index('overalls'):
            neurodoll_whole_body_index = multilabel_to_ultimate21_conversion[whole_body_winner_index]
            n = len(final_mask[final_mask==neurodoll_wholebody_index])
            logging.info('n in final mask from wholebody (overall) alone:'+str(n))
            for i in upper_cover_indexlist:
                pass  #upper cover ok with overalls
            for i in upper_under_indexlist:
                pass #upper under ok with overalls
            for i in lower_cover_indexlist: #lower cover donated to overalls
                nd_index = multilabel_to_ultimate21_conversion[i]
                if nd_index is None:
                    logging.warning('ml index {} has no conversion '.format(i))
                    continue
                final_mask[final_mask==nd_index] = neurodoll_wholebody_index
                logging.info('uppercover nd index {} donated to overalls'.format(nd_index))
                n = len(final_mask[final_mask==neurodoll_wholebody_index])
                logging.info('n in final mask from wholebody alone:'+str(n))
            for i in lower_under_indexlist: #lower under donated to overalls - this can conceivably go wrong e.g. with short overalls and stockings
                nd_index = multilabel_to_ultimate21_conversion[i]
                if nd_index is None:
                    logging.warning('ml index {} has no conversion '.format(i))
                    continue
                final_mask[final_mask==nd_index] = neurodoll_wholebody_index
                logging.info('uppercover nd index {} donated to overalls'.format(nd_index))
                n = len(final_mask[final_mask==neurodoll_wholebody_index])
                logging.info('n in final mask from wholebody alone:'+str(n))
#not overalls, so donate  whole_body to upper_under - maybe not to  lower_under . Not clear what to do actually.
        else: #not overalls
            if upper_winner_nd_index is None:
                logging.warning('nd wholebody index {} ml index {} has no conversion '.format(upper_winner_nd_index,upper_winner_index))
            else:
                n1 = len(final_mask[final_mask==upper_winner_nd_index])
                n2 = len(final_mask[final_mask==lower_winner_nd_index])
                logging.info('n in final mask from wholebody before donation to upper winner {} px,nd {} (lower {} px,nd {}'.format(n1,upper_winner_nd_index,n2,lower_winner_nd_index))
                #todo - actually only wholebody pixels in the upper half of the image should be donated
                for i in whole_body_indexlist: #whole_body donated to upper_under
                    nd_index = multilabel_to_ultimate21_conversion[i]
                    if nd_index is None:
                        logging.warning('ml index {} has no conversion (4upper)'.format(i))
                        continue            #donate upper pixels to upper_winner
                    logging.debug('3. donating nd {} in top of wholebody to upper_under and bottom to lower_under'.format(nd_index))
                    logging.debug('first adding from upper part of nd {}  to nd {}, ysplit {}'.format(nd_index,upper_winner_nd_index,y_split))
                    for y in range(0, final_mask.shape[0]):
                        if y <= y_split:
                            for x in range(0, final_mask.shape[1]):
                                if final_mask[y][x] == nd_index:
                                    final_mask[y][x] = upper_winner_nd_index
                n1 = len(final_mask[final_mask==upper_winner_nd_index])
                n2 = len(final_mask[final_mask==lower_winner_nd_index])
                logging.info('n in final mask from wholebody after donation to upper winner {} px,nd {} (lower {} px,nd {}'.format(n1,upper_winner_nd_index,n2,lower_winner_nd_index))

    # donate whole-body pixels to lower winner
            if lower_winner_nd_index is None:
                logging.warning('nd wholebody index {} ml index {} has no conversion '.format(upper_winner_nd_index,upper_winner_index))
            else:
                n1 = len(final_mask[final_mask==upper_winner_nd_index])
                n2 = len(final_mask[final_mask==lower_winner_nd_index])
                logging.info('n in final mask from wholebody after donation to upper winner {} px,nd {} , lower winner {} px, nd {} '.format(n1,upper_winner_nd_index,n2,lower_winner_nd_index))
                #todo - actually only wholebody pixels in the upper half of the image should be donated
                for i in whole_body_indexlist: #whole_body donated to upper_under
                    nd_index = multilabel_to_ultimate21_conversion[i]
                    if nd_index is None:
                        logging.warning('ml index {} has no conversion (4lower)'.format(i))
                        continue
            #donate upper pixels to upper_winner
                    logging.debug('3. donating nd {} in botto of wholebody to upper_under and bottom to lower_under'.format(nd_index))
                    logging.debug('second, adding from lower part of nd {}  to nd {}, ysplit {}'.format(nd_index,lower_winner_nd_index,y_split))
                    for y in range(0, final_mask.shape[0]):
                        if y > y_split:
                            for x in range(0, final_mask.shape[1]):
                                if final_mask[y][x] == nd_index:
                                    final_mask[y][x] = lower_winner_nd_index
            #donate upper pixels to lower_winner
                n1 = len(final_mask[final_mask==upper_winner_nd_index])
                n2 = len(final_mask[final_mask==lower_winner_nd_index])
                logging.info('n in final mask from wholebody donation to upper {} and lower {}:'.format(n1,n2))



        logging.debug('after case two pixel values look like')
        count_values(final_mask,labels=constants.ultimate_21)

# third case - lowercover > wholebody > upper_under
# here its not clear who to sack - the lowercover or the wholebody
# so i arbitrarily decided to sack the whole_body in favor of the lowercover since lowercover is higher
# donate lower part of wholebody to lowerwinner and upper part to upper winner
# this can be combined with second case I guess as there is nothing different - whole body gets added to lower/upper winners

    elif (whole_body_winner_value<lower_cover_winner_value) and (whole_body_winner_value>upper_under_winner_value) and whole_body_winner_value>multilabel_threshold:
        logging.info('case 3. one part {} > upper under {} and < lower cover {}'.format(whole_body_winner_value,upper_under_winner_value,lower_cover_winner_value))
        if upper_winner_nd_index is None:
            logging.warning('nd wholebody index {} ml index {} has no conversion '.format(upper_winner_nd_index,upper_winner_index))
        else:
            n1 = len(final_mask[final_mask==upper_winner_nd_index])
            n2 = len(final_mask[final_mask==lower_winner_nd_index])
            logging.info('n in final mask from wholebody before donation to upper winner {} px,nd {} (lower {} px,nd {}'.format(n1,upper_winner_nd_index,n2,lower_winner_nd_index))
            #todo - actually only wholebody pixels in the upper half of the image should be donated
            for i in whole_body_indexlist: #whole_body donated to upper_under
                nd_index = multilabel_to_ultimate21_conversion[i]
                if nd_index is None:
                    logging.warning('ml index {} has no conversion (4upper)'.format(i))
                    continue            #donate upper pixels to upper_winner
                logging.debug('3. donating nd {} in top of wholebody to upper_under and bottom to lower_under'.format(nd_index))
                logging.debug('first adding from upper part of nd {}  to nd {}, ysplit {}'.format(nd_index,upper_winner_nd_index,y_split))
                for y in range(0, final_mask.shape[0]):
                    if y <= y_split:
                        for x in range(0, final_mask.shape[1]):
                            if final_mask[y][x] == nd_index:
                                final_mask[y][x] = upper_winner_nd_index
            n1 = len(final_mask[final_mask==upper_winner_nd_index])
            n2 = len(final_mask[final_mask==lower_winner_nd_index])
            logging.info('n in final mask from wholebody after donation to upper winner {} px,nd {} (lower {} px,nd {}'.format(n1,upper_winner_nd_index,n2,lower_winner_nd_index))

# donate whole-body pixels to lower winner
        if lower_winner_nd_index is None:
            logging.warning('nd wholebody index {} ml index {} has no conversion '.format(upper_winner_nd_index,upper_winner_index))
        else:
            n1 = len(final_mask[final_mask==upper_winner_nd_index])
            n2 = len(final_mask[final_mask==lower_winner_nd_index])
            logging.info('n in final mask from wholebody after donation to upper winner {} px,nd {} , lower winner {} px, nd {} '.format(n1,upper_winner_nd_index,n2,lower_winner_nd_index))
            #todo - actually only wholebody pixels in the upper half of the image should be donated
            for i in whole_body_indexlist: #whole_body donated to upper_under
                nd_index = multilabel_to_ultimate21_conversion[i]
                if nd_index is None:
                    logging.warning('ml index {} has no conversion (4lower)'.format(i))
                    continue
        #donate upper pixels to upper_winner
                logging.debug('3. donating nd {} in botto of wholebody to upper_under and bottom to lower_under'.format(nd_index))
                logging.debug('second, adding from lower part of nd {}  to nd {}, ysplit {}'.format(nd_index,lower_winner_nd_index,y_split))
                for y in range(0, final_mask.shape[0]):
                    if y > y_split:
                        for x in range(0, final_mask.shape[1]):
                            if final_mask[y][x] == nd_index:
                                final_mask[y][x] = lower_winner_nd_index
        #donate upper pixels to lower_winner
            n1 = len(final_mask[final_mask==upper_winner_nd_index])
            n2 = len(final_mask[final_mask==lower_winner_nd_index])
            logging.info('n in final mask from wholebody donation to upper {} and lower {}:'.format(n1,n2))

# fourth case - lowercover , upper_under > wholebody
# sack wholebody in favor of upper and lower
# donate top of wholebody to greater of upper cover/upper under (yes this is arbitrary and possibly wrong)
# donate bottom pixels of wholebody to greater of lower cover/lower under (again somewhat arbitrary)
# this also could get combined with #2,3 I suppose
# neurodoll_upper_cover_index = multilabel_to_ultimate21_conversion[upper_cover_winner_index] #
        logging.debug('after case three pixel values look like')
        count_values(final_mask,labels=constants.ultimate_21)


    elif (whole_body_winner_value<lower_cover_winner_value) and (whole_body_winner_value<upper_under_winner_value):
        logging.info('case 4.one part {} < upper under {} and < lower cover {}'.format(whole_body_winner_value,upper_under_winner_value,lower_cover_winner_value))
        neurodoll_lower_cover_index = multilabel_to_ultimate21_conversion[lower_cover_winner_index]
# donate whole-body pixels to upper winner
        if upper_winner_nd_index is None:
            logging.warning('nd wholebody index {} ml index {} has no conversion '.format(upper_winner_nd_index,upper_winner_index))
        else:
            n1 = len(final_mask[final_mask==upper_winner_nd_index])
            n2 = len(final_mask[final_mask==lower_winner_nd_index])
            logging.info('n in final mask from wholebody before donation to upper winner {} px,nd {} (lower {} px,nd {}'.format(n1,upper_winner_nd_index,n2,lower_winner_nd_index))
            #todo - actually only wholebody pixels in the upper half of the image should be donated
            for i in whole_body_indexlist: #whole_body donated to upper_under
                nd_index = multilabel_to_ultimate21_conversion[i]
                if nd_index is None:
                    logging.warning('ml index {} has no conversion (4upper)'.format(i))
                    continue            #donate upper pixels to upper_winner
                logging.debug('4. donating nd {} in top of wholebody to upper_under and bottom to lower_under'.format(nd_index))
                logging.debug('first adding from upper part of nd {}  to nd {}, ysplit {}'.format(nd_index,upper_winner_nd_index,y_split))
                for y in range(0, final_mask.shape[0]):
                    if y <= y_split:
                        for x in range(0, final_mask.shape[1]):
                            if final_mask[y][x] == nd_index:
                                final_mask[y][x] = upper_winner_nd_index
            n1 = len(final_mask[final_mask==upper_winner_nd_index])
            n2 = len(final_mask[final_mask==lower_winner_nd_index])
            logging.info('n in final mask from wholebody after donation to upper winner {} px,nd {} (lower {} px,nd {}'.format(n1,upper_winner_nd_index,n2,lower_winner_nd_index))

# donate whole-body pixels to lower winner
        if lower_winner_nd_index is None:
            logging.warning('nd wholebody index {} ml index {} has no conversion '.format(upper_winner_nd_index,upper_winner_index))
        else:
            n1 = len(final_mask[final_mask==upper_winner_nd_index])
            n2 = len(final_mask[final_mask==lower_winner_nd_index])
            logging.info('n in final mask from wholebody after donation to upper winner {} px,nd {} , lower winner {} px, nd {} '.format(n1,upper_winner_nd_index,n2,lower_winner_nd_index))
            #todo - actually only wholebody pixels in the upper half of the image should be donated
            for i in whole_body_indexlist: #whole_body donated to upper_under
                nd_index = multilabel_to_ultimate21_conversion[i]
                if nd_index is None:
                    logging.warning('ml index {} has no conversion (4lower)'.format(i))
                    continue
        #donate upper pixels to upper_winner
                logging.debug('4. donating nd {} in botto of wholebody to upper_under and bottom to lower_under'.format(nd_index))
                logging.debug('second, adding from lower part of nd {}  to nd {}, ysplit {}'.format(nd_index,lower_winner_nd_index,y_split))
                for y in range(0, final_mask.shape[0]):
                    if y > y_split:
                        for x in range(0, final_mask.shape[1]):
                            if final_mask[y][x] == nd_index:
                                final_mask[y][x] = lower_winner_nd_index
        #donate upper pixels to lower_winner
            n1 = len(final_mask[final_mask==upper_winner_nd_index])
            n2 = len(final_mask[final_mask==lower_winner_nd_index])
            logging.info('n in final mask from wholebody donation to upper {} and lower {}:'.format(n1,n2))

        logging.debug('after case four pixel values look like')
        count_values(final_mask,labels=constants.ultimate_21)

    foreground = np.array((pixlevel_categorical_output>0)*1)  #*1 turns T/F into 1/0
    final_mask = final_mask * foreground # only keep stuff that was part of original fg - this is already  true
    # unless we start adding pixvalues that didn't 'win'

    #7. if no lower cover and no whole-body was decided upon above: take max of lowercover items , donate losers to winner
    #8. take at most one lower under, donate losers to winner

    if(0):
        for i in range(len(thresholded_multilabel)):
            if thresholded_multilabel[i]:
                neurodoll_index = multilabel_to_ultimate21_conversion[i]
                if neurodoll_index is None:
                    print('no mapping from index {} (label {}) to neurodoll'.format(i,multilabel_labels[i]))
                    continue
                nd_pixels = len(pixlevel_categorical_output[pixlevel_categorical_output==neurodoll_index])
                print('index {} webtoollabel {} newindex {} neurodoll_label {} was above threshold {} (ml value {}) nd_pixels {}'.format(
                    i,multilabel_labels[i],neurodoll_index,constants.ultimate_21[neurodoll_index], multilabel_threshold,multilabel[i],nd_pixels))
                gray_layer = graylevel_nd_output[:,:,neurodoll_index]
                print('gray layer size:'+str(gray_layer.shape))
    #            item_mask = grabcut_using_neurodoll_output(url_or_np_array,neurodoll_index,median_factor=median_factor)
                if nd_pixels>0:  #possibly put a threshold here, too few pixels and forget about it
                    item_mask = grabcut_using_neurodoll_graylevel(url_or_np_array,gray_layer,median_factor=median_factor)
                    #the grabcut results dont seem too hot so i am moving to a 'nadav style' from-nd-and-ml-to-results system
                #namely : for top , decide if its a top or dress or jacket
                # for bottom, decide if dress/pants/skirt
                    pass
                else:
                    print('no pixels in mask, skipping')
                if item_mask is None:
                    continue
                item_mask = np.multiply(item_mask,neurodoll_index)
                if first_time_thru:
                    final_mask = np.zeros_like(item_mask)
                    first_time_thru = False
                unique_to_new_mask = np.logical_and(item_mask != 0,final_mask == 0)   #dealing with same pixel claimed by two masks. if two masks include same pixel take first, don't add the pixel vals together
                unique_to_new_mask = np.multiply(unique_to_new_mask,neurodoll_index)
                final_mask = final_mask + unique_to_new_mask
    #            cv2.imshow('mask '+str(i),item_mask)
    #            cv2.waitKey(0)
    timestamp = int(10*time.time())

    #write file (for debugging)
    name = orig_filename+'_combinedoutput.png'

    print('combined png name:'+name+' orig filename '+orig_filename)
    cv2.imwrite(name,final_mask)
    nice_output = imutils.show_mask_with_labels(name,constants.ultimate_21,save_images=True,original_image=orig_filename+'.jpg',visual_output=False)
#    nice_output = imutils.show_mask_with_labels(name,constants.ultimate_21,save_images=True)

    #save graymask, this should be identical to nd except no threshold on low amt of pixels
    graymask_filename = orig_filename+'_origmask.png'
    print('original mask file:'+graymask_filename)
    cv2.imwrite(graymask_filename,pixlevel_categorical_output)
    nice_output = imutils.show_mask_with_labels(graymask_filename,constants.ultimate_21,save_images=True,original_image=orig_filename+'.jpg',visual_output=False)
    count_values(final_mask,labels=constants.ultimate_21)

    return final_mask
