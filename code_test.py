__author__ = 'Nadav Paz'

from Tkinter import Tk
from tkFileDialog import askopenfilename
import itertools

import cv2
import numpy as np
from bson import json_util

import background_removal
import constants
import Utils
import kassper
import fingerprint
import NNSearch


def gc2mask_test(image, bb):
    small_image, resize_ratio = background_removal.standard_resize(image, 400)    # shrink image for faster process
    bb = np.array(bb)/resize_ratio
    bb = bb.astype(np.uint16)                                                     # shrink bb in the same ratio
    # bb = [int(b) for b in (np.array(bb)/resize_ratio)]
    x, y, w, h = bb
    cv2.rectangle(small_image, (x, y), (x+w, y+h), [0, 255, 0], 2)
    cv2.imshow('1', small_image)
    cv2.waitKey(0)
    fg_mask = background_removal.get_fg_mask(small_image, bb)                     # returns the grab-cut mask (if bb => PFG-PBG gc, if !bb => face gc)
    cv2.imshow('2', background_removal.get_masked_image(small_image, fg_mask))
    cv2.waitKey(0)
    bb_mask = background_removal.get_binary_bb_mask(small_image, bb)                     # bounding box mask
    cv2.imshow('3', background_removal.get_masked_image(small_image, bb_mask))
    cv2.waitKey(0)
    combined_mask = cv2.bitwise_and(fg_mask, bb_mask)                             # for sending the right mask to the fp
    cv2.imshow('4', background_removal.get_masked_image(small_image, combined_mask))
    cv2.waitKey(0)
    return


def get_image():
    Tk().withdraw()
    filename = askopenfilename()
    big_image = cv2.imread(filename)
    return big_image


def lomshane_test():
    """inputs = [json_util.loads(
        '{"url":"http://msc.wcdn.co.il/w/w-635/1684386-5.jpg","bb":"[137.2972972972973,188.80597014925374,356.97297297297297,319.2537313432836]","keyword":"mens-outerwear","post_id":"552a79359e31f134f0f9c401"}')
    , json_util.loads(
        '{"url":"http://msc.wcdn.co.il/w/w-635/1684386-5.jpg","bb":"[50.2972972972973,50.80597014925374,70.97297297297297,70.2537313432836]","keyword":"mens-outerwear","post_id":"552a79359e31f134f0f9c401"}')
    ,json_util.loads(
        '{"url":"http://msc.wcdn.co.il/w/w-635/1684386-5.jpg","bb":"[0, 0, 100.97297297297297, 100.2537313432836]","keyword":"mens-outerwear","post_id":"552a79359e31f134f0f9c401"}')
    ]"""
    bbs = [[9, 135, 97, 87], [200, 299, 98, 67], [9, 272, 120, 95], [316, 13, 83, 138]]
    image, ratio = background_removal.standard_resize(get_image(), 400)
    masks = [make_mask_test(image, bb) for bb in bbs]
    fingers = [fingerprint.fp(image, mask) for mask in masks]
    print np.array([(pair[0] - pair[1]) for pair in itertools.combinations(fingers, 2)])
    cv2.imshow('mask1', masks[0])
    cv2.imshow('mask2', masks[1])
    cv2.imshow('mask3', masks[2])
    cv2.imshow('mask4', masks[3])
    cv2.waitKey(0)
    return


def make_mask_test(image_url, bb=None):
    svg_address = constants.svg_folder
    image = Utils.get_cv2_img_array(image_url)  # turn the URL into a cv2 image
    small_image, resize_ratio = background_removal.standard_resize(image, 400)  # shrink image for faster process
    bb = [int(b) for b in (np.array(bb) / resize_ratio)]  # shrink bb in the same ratio
    fg_mask = background_removal.get_fg_mask(small_image,
                                             bb)  # returns the grab-cut mask (if bb => PFG-PBG gc, if !bb => face gc)
    # bb_mask = background_removal.get_binary_bb_mask(small_image, bb)            # bounding box mask
    # combined_mask = cv2.bitwise_and(fg_mask, bb_mask)                           # for sending the right mask to the fp
    gc_image = background_removal.get_masked_image(small_image, fg_mask)
    face_rect = background_removal.find_face(small_image)
    if len(face_rect) > 0:
        x, y, w, h = face_rect[0]
        face_image = image[y:y + h, x:x + w, :]
        without_skin = kassper.skin_removal(face_image, gc_image)
        crawl_mask = kassper.clutter_removal(without_skin, 200)
        without_clutter = background_removal.get_masked_image(without_skin, crawl_mask)
        mask = kassper.get_mask(without_clutter)
    else:
        mask = kassper.get_mask(gc_image)
    return mask


def hadasha_test():
    results = json_util.loads(
        '{"matches": [{"fingerPrintVector": [0.46159881353378296, 0.5813536047935486, 0.4549544155597687, -1.3243932723999023, -1.44627845287323, -1.651790738105774, 0.011443973518908024, 0.46668434143066406, 0.49346059560775757, 0.000530222721863538, 0.001060445443727076, 0.0003976670268457383, 8.83704487932846e-05, 4.41852243966423e-05, 0.0, 0.0002209261292591691, 8.83704487932846e-05, 8.83704487932846e-05, 0.0001767408975865692, 0.0001767408975865692, 0.00030929656350053847, 0.001060445443727076, 0.0007953340536914766, 0.001369741978123784, 0.001369741978123784, 0.0014139271806925535, 0.0009720749221742153, 0.0013255567755550146, 0.0013255567755550146, 0.011974195949733257, 0.0036231884732842445, 0.0002209261292591691, 0.0009278897196054459, 0.0018115942366421223, 0.003004595171660185, 0.013918345794081688, 0.00459526339545846, 0.0025627429131418467, 0.0030487803742289543, 0.0029604099690914154, 0.0024743725080043077, 0.0017232237150892615, 0.001369741978123784, 0.0014139271806925535, 0.0012813714565709233, 0.0018557794392108917, 0.0053022271022200584, 0.010250971652567387, 0.045334041118621826, 0.748939573764801, 0.13379286229610443, 0.008837045170366764, 0.0020325202494859695, 0.001060445443727076, 0.0004418522585183382, 8.83704487932846e-05], "clothingClass": "evening-dresses", "imageURL": "http://resources.shopstyle.com/sim/ad/31/ad31f8bd63eaa505178847de08347cde/agnona-two-tone-silk-gown.jpg", "id": 461335843, "buyURL": "http://www.shopstyle.com/action/apiVisitRetailer?id=461335843&pid=uid900-25284470-95"}, {"fingerPrintVector": [0.9166110157966614, 0.49660155177116394, 0.2840794324874878, -0.299735963344574, -1.4207818508148193, -2.258920669555664, 0.9568701386451721, 0.028497770428657532, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0005574136157520115, 0.01407469343394041, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0008361203945241868, 0.0025083611253648996, 0.00264771468937397, 0.010799888521432877, 0.0627090334892273, 0.6639492511749268, 0.22561316192150116, 0.028358416631817818, 0.0020206242334097624, 0.0005574136157520115, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "clothingClass": "evening-dresses", "imageURL": "http://resources.shopstyle.com/sim/52/72/5272feb97bb281258dc0ef67f1c2d9a9/aglini-short-dresses.jpg", "id": 451888603, "buyURL": "http://www.shopstyle.com/action/apiVisitRetailer?id=451888603&pid=uid900-25284470-95"}, {"fingerPrintVector": [0.6156277060508728, 0.2835935652256012, 0.28309088945388794, -1.321216106414795, -2.4193997383117676, -2.146038055419922, 0.11269468069076538, 0.04902536794543266, 0.019933391362428665, 0.0015182681381702423, 0.003281418466940522, 0.0021059848368167877, 0.0010774806141853333, 0.0012244097888469696, 0.0006366930902004242, 0.0019100793870165944, 0.0012244097888469696, 0.013419532217085361, 0.7745127081871033, 0.0018611029954627156, 0.0005877166986465454, 0.0007346458733081818, 0.0005387403070926666, 0.0003428347408771515, 0.00014692917466163635, 0.0006366930902004242, 0.00019590558076743037, 0.0012733861804008484, 0.0023998431861400604, 0.0038691351655870676, 0.004848662763834, 0.00039181116153486073, 0.0002448819577693939, 0.0033793712500482798, 0.009550396353006363, 0.015231658704578876, 0.0690077394247055, 0.05176804959774017, 0.01611323282122612, 0.027524733915925026, 0.004799686372280121, 0.004946615546941757, 0.005926143378019333, 0.01812126487493515, 0.02213732898235321, 0.3416103422641754, 0.39641493558883667, 0.00715055363252759, 0.0018121266039088368, 0.0008325987146236002, 0.0004407875530887395, 0.0004897639155387878, 0.0005387403070926666, 0.0002448819577693939, 0.0002938583493232727, 0.0], "clothingClass": "evening-dresses", "imageURL": "http://resources.shopstyle.com/sim/23/0a/230aaa1d37c478c9f39547d9fb42b514/agnona-color-block-silk-crepe-maxi-dress.jpg", "id": 458165229, "buyURL": "http://www.shopstyle.com/action/apiVisitRetailer?id=458165229&pid=uid900-25284470-95"}, {"fingerPrintVector": [0.27502569556236267, 0.2403670698404312, 0.3831014037132263, -2.429974317550659, -2.4978291988372803, -1.6643863916397095, 0.05078022554516792, 0.0, 0.01167060062289238, 0.0, 0.009965905919671059, 0.00026226069894619286, 0.00147521635517478, 0.0, 0.0, 0.0, 0.03717545419931412, 0.016194596886634827, 0.20548124611377716, 0.45810386538505554, 0.12204957008361816, 0.056156568229198456, 0.012752425856888294, 0.0, 0.0, 0.0015407815808430314, 0.0, 0.0, 0.016391292214393616, 0.0, 0.0, 0.04953448846936226, 0.0010490427957847714, 0.025898242369294167, 0.05681222304701805, 0.2159716784954071, 0.39555469155311584, 0.16437189280986786, 0.06038552150130272, 0.013768685981631279, 0.0035077366046607494, 0.003704432165250182, 0.0006884342874400318, 0.004687909968197346, 0.0010490427957847714, 0.0, 0.001606346690095961, 0.0005573039525188506, 0.00026226069894619286, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "clothingClass": "evening-dresses", "imageURL": "http://resources.shopstyle.com/sim/b4/dd/b4dd42c28494c8318745aa93a6285bb4/agnona-loose-fit-dress.jpg", "id": 459813391, "buyURL": "http://www.shopstyle.com/action/apiVisitRetailer?id=459813391&pid=uid900-25284470-95"}, {"fingerPrintVector": [0.8230574131011963, 0.2422187626361847, 0.24179038405418396, -0.522312343120575, -2.5765397548675537, -2.6008198261260986, 0.9029062986373901, 0.08826278150081635, 0.004765909630805254, 0.00037379685090854764, 0.00032707222271710634, 0.00014017381181474775, 4.6724606363568455e-05, 0.0, 0.0, 4.6724606363568455e-05, 4.6724606363568455e-05, 0.0, 0.00032707222271710634, 4.6724606363568455e-05, 4.6724606363568455e-05, 4.6724606363568455e-05, 0.00014017381181474775, 4.6724606363568455e-05, 0.0, 0.00014017381181474775, 0.00014017381181474775, 4.6724606363568455e-05, 0.0, 4.6724606363568455e-05, 0.0020558827091008425, 0.002710027154535055, 0.001541911973617971, 0.03789365664124489, 0.40673768520355225, 0.22726847231388092, 0.09153350442647934, 0.08541257679462433, 0.08335669338703156, 0.030884964391589165, 0.0050929817371070385, 0.004672460723668337, 0.0034576207399368286, 0.003223997773602605, 0.002990374807268381, 0.003364171599969268, 0.003364171599969268, 0.002102607162669301, 0.001682085799984634, 0.0009344920981675386, 0.0004672460490837693, 0.000700869073625654, 0.00032707222271710634, 0.00014017381181474775, 0.00014017381181474775, 0.0], "clothingClass": "evening-dresses", "imageURL": "http://resources.shopstyle.com/sim/a1/20/a120cc39e417af290a2d028c7ee5e43a/aidan-mattox-lace-illusion-gown-with-sweetheart-neckline.jpg", "id": 464614505, "buyURL": "http://www.shopstyle.com/action/apiVisitRetailer?id=464614505&pid=uid900-25284470-95"}, {"fingerPrintVector": [0.19684778153896332, 0.22092501819133759, 0.15789924561977386, -2.7879648208618164, -2.716064214706421, -3.109351873397827, 0.31328585743904114, 0.11051356047391891, 0.009922739118337631, 0.003383325645700097, 0.00573145505040884, 0.0014644245384261012, 0.0018179063918069005, 0.00030298440833576024, 2.5248698875657283e-05, 0.000530222721863538, 0.004317527636885643, 0.0007069635903462768, 0.00030298440833576024, 0.002171388128772378, 0.0026006160769611597, 0.009897490032017231, 0.013331313617527485, 0.07387769222259521, 0.21617937088012695, 0.18171489238739014, 0.025652678683400154, 0.004620512016117573, 0.010730697773396969, 0.0026511135511100292, 0.004267030395567417, 0.3632025420665741, 0.021158410236239433, 0.11016007512807846, 0.2520325183868408, 0.09490986168384552, 0.036459121853113174, 0.026738373562693596, 0.030197445303201675, 0.02353178896009922, 0.01492198184132576, 0.010932686738669872, 0.005024491343647242, 0.0031560873612761497, 0.002045144559815526, 0.0017674090340733528, 0.001489673275500536, 0.0007322122692130506, 0.0004797253059223294, 0.0003534817951731384, 0.0001767408975865692, 0.0001767408975865692, 5.0497397751314566e-05, 0.00015149220416788012, 5.0497397751314566e-05, 2.5248698875657283e-05], "clothingClass": "evening-dresses", "imageURL": "http://resources.shopstyle.com/sim/33/e7/33e7da861c8cfb6504bb31716ce90ca6/aidan-mattox-aidan-by-sequin-knit-ball-gown.jpg", "id": 464510620, "buyURL": "http://www.shopstyle.com/action/apiVisitRetailer?id=464510620&pid=uid900-25284470-95"}, {"fingerPrintVector": [0.3072342872619629, 0.2105940878391266, 0.08130573481321335, -2.414323329925537, -2.8246397972106934, -3.911099433898926, 0.1023000106215477, 0.06628333032131195, 0.0010109945433214307, 0.0001684991002548486, 0.0009056826238520443, 0.00025274863583035767, 0.0004633725038729608, 4.212477506371215e-05, 0.0010952440788969398, 0.0001474367018090561, 0.0007582459365949035, 0.00037912296829745173, 0.00035806058440357447, 0.0009688697755336761, 0.0015375541988760233, 0.005391971208155155, 0.012468933127820492, 0.5173975229263306, 0.13134504854679108, 0.06523021310567856, 0.048654112964868546, 0.01990395598113537, 0.012342558242380619, 0.003728042356669903, 0.006866338197141886, 0.013290366157889366, 0.0016428661765530705, 0.010720754973590374, 0.2492522895336151, 0.3590715825557709, 0.10885041207075119, 0.032478202134370804, 0.025169551372528076, 0.027760226279497147, 0.04157715290784836, 0.04591600224375725, 0.02700198069214821, 0.016344411298632622, 0.007477147504687309, 0.002527486300095916, 0.003917603753507137, 0.003201482817530632, 0.0017903029220178723, 0.0023800497874617577, 0.0005686844233423471, 0.0014322423376142979, 0.000800370704382658, 0.0005054972716607153, 0.00027381101972423494, 0.00010531193402130157], "clothingClass": "evening-dresses", "imageURL": "http://resources.shopstyle.com/sim/b1/10/b1104074a317dee7d3b6bee4a04e9895/agent-provocateur-mareko-dress.jpg", "id": 458147701, "buyURL": "http://www.shopstyle.com/action/apiVisitRetailer?id=458147701&pid=uid900-25284470-95"}, {"fingerPrintVector": [0.27039071917533875, 0.19728915393352509, 0.08452802896499634, -2.5963382720947266, -3.0069587230682373, -3.7787938117980957, 0.06552508473396301, 0.07658284157514572, 0.0014533046633005142, 0.00021062386804260314, 0.0003159357875119895, 0.00010531193402130157, 0.0001684991002548486, 6.318715895758942e-05, 0.00027381101972423494, 8.42495501274243e-05, 0.000631871575023979, 0.0001684991002548486, 4.212477506371215e-05, 0.00037912296829745173, 0.0009688697755336761, 0.0048864735290408134, 0.03557437285780907, 0.4842032194137573, 0.10183663666248322, 0.08486035466194153, 0.003917603753507137, 0.06044904887676239, 0.016597161069512367, 0.006613589357584715, 0.05408820882439613, 0.00400185352191329, 0.003433169098570943, 0.03751211240887642, 0.17372256517410278, 0.3717300593852997, 0.1418762356042862, 0.03761742264032364, 0.029087156057357788, 0.03150933235883713, 0.027760226279497147, 0.0357428714632988, 0.02645435743033886, 0.029002906754612923, 0.010046758688986301, 0.004970723297446966, 0.019103584811091423, 0.0026538607198745012, 0.0019798644352704287, 0.003980791196227074, 0.0007793083204887807, 0.0010952440788969398, 0.00035806058440357447, 0.00021062386804260314, 0.00018956148414872587, 4.212477506371215e-05], "clothingClass": "evening-dresses", "imageURL": "http://resources.shopstyle.com/sim/2e/92/2e9244dcc23b3459768dd914b323ee38/agent-provocateur-luna-long-slip.jpg", "id": 434040771, "buyURL": "http://www.shopstyle.com/action/apiVisitRetailer?id=434040771&pid=uid900-25284470-95"}, {"fingerPrintVector": [0.22861936688423157, 0.12705032527446747, 0.07096967101097107, -2.623910427093506, -2.7965164184570312, -4.118924140930176, 0.08652428537607193, 0.05790049955248833, 0.0015796789666637778, 0.00018956148414872587, 0.0006950587849132717, 0.00010531193402130157, 8.42495501274243e-05, 2.1062387531856075e-05, 0.0014322423376142979, 0.0001474367018090561, 0.00018956148414872587, 0.00010531193402130157, 0.00025274863583035767, 0.14402459561824799, 0.1870129257440567, 0.012405745685100555, 0.010404818691313267, 0.3960149884223938, 0.0687897577881813, 0.011289439164102077, 0.007814145646989346, 0.002632798394188285, 0.003433169098570943, 0.001916677225381136, 0.005033910274505615, 0.0034121067728847265, 0.0019588018767535686, 0.007161211688071489, 0.1936475783586502, 0.2492101639509201, 0.048211801797151566, 0.03273094817996025, 0.03249926120042801, 0.026433294638991356, 0.022642066702246666, 0.0159442275762558, 0.008024768903851509, 0.0040861028246581554, 0.002717047929763794, 0.001916677225381136, 0.002801297465339303, 0.002148363506421447, 0.0018745524575933814, 0.0022747376933693886, 0.0022115507163107395, 0.0037912295665591955, 0.004423101432621479, 0.007266523316502571, 0.017671342939138412, 0.144719660282135], "clothingClass": "evening-dresses", "imageURL": "http://resources.shopstyle.com/sim/11/75/117589d5d4563fe71a60ee6de4d70dfb/agent-provocateur-cassia-slip.jpg", "id": 459424002, "buyURL": "http://www.shopstyle.com/action/apiVisitRetailer?id=459424002&pid=uid900-25284470-95"}, {"fingerPrintVector": [0.17680856585502625, 0.06432288885116577, 0.06566708534955978, -2.94968318939209, -3.9931600093841553, -4.125364780426025, 0.28531110286712646, 0.20548464357852936, 0.003517418634146452, 0.0006108091911301017, 0.0017271157121285796, 0.0002316862519364804, 0.0002316862519364804, 6.318715895758942e-05, 0.0009478073916397989, 6.318715895758942e-05, 0.0014954294310882688, 0.00035806058440357447, 0.00021062386804260314, 0.0005476220394484699, 0.0010952440788969398, 0.005118160042911768, 0.012363621033728123, 0.20141960680484772, 0.044357385486364365, 0.03755423426628113, 0.030287712812423706, 0.02205231972038746, 0.03696448728442192, 0.03926029056310654, 0.0687265694141388, 0.010594381019473076, 0.0010531193111091852, 0.014153923839330673, 0.11957117170095444, 0.1100299060344696, 0.03538481146097183, 0.03593243286013603, 0.04338851571083069, 0.05455158278346062, 0.06390327960252762, 0.0847339853644371, 0.07856269925832748, 0.0775306448340416, 0.05301402881741524, 0.03523737192153931, 0.03338388353586197, 0.02409537136554718, 0.01666034758090973, 0.01590210199356079, 0.010826067067682743, 0.013100804761052132, 0.010194195434451103, 0.007119086571037769, 0.005707907024770975, 0.0018534900154918432], "clothingClass": "evening-dresses", "imageURL": "http://resources.shopstyle.com/sim/4e/26/4e26c9e05187af25d0423c0021c6f29f/agent-provocateur-denver-gown.jpg", "id": 456356178, "buyURL": "http://www.shopstyle.com/action/apiVisitRetailer?id=456356178&pid=uid900-25284470-95"}], "fingerPrintVector": [0.8329296708106995, 0.24874667823314667, 0.14163444936275482, 0.5071839094161987, 2.6389594078063965, 3.3159875869750977, 0.9096829891204834, 0.07101380825042725, 7.846829976188019e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00015693659952376038, 0.0190677959471941, 0.00039234149153344333, 0.007611425127834082, 0.011848713271319866, 0.011848713271319866, 0.005100439302623272, 0.007219083607196808, 0.011770244687795639, 0.016242938116192818, 0.032093532383441925, 0.03640929237008095, 0.05171060934662819, 0.10491211712360382, 0.41541117429733276, 0.24050533771514893, 0.03790018707513809, 0.00643440056592226, 0.0019617073703557253, 0.00039234149153344333, 0.00023540489200968295, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "svg": "http://extremeli.trendi.guru/static/svgs/552c09029e31f16d6c89d3af.svg"}')
    fp0 = results["matches"][0]["fingerPrintVector"]
    fp1 = results["matches"][1]["fingerPrintVector"]
    fp2 = results["matches"][2]["fingerPrintVector"]

    fp = results["fingerPrintVector"]
    d0 = NNSearch.distance_1_k(fp0, fp, 1.5)
    d1 = NNSearch.distance_1_k(fp1, fp, 1.5)
    d2 = NNSearch.distance_1_k(fp2, fp, 1.5)

    print d0, d1, d2


def skin_removal_test():
    image, ratio = background_removal.standard_resize(background_removal.get_image(), 400)
    fg_mask = background_removal.get_fg_mask(image)
    gc_image = background_removal.get_masked_image(image, fg_mask)
    face_rect = background_removal.find_face(image)
    x, y, w, h = face_rect[0]
    face_image = image[x:x + w, y:y + h, :]
    without_skin = kassper.skin_removal(gc_image, face_image)
    cv2.imshow('original', image)
    cv2.imshow('gc', gc_image)
    cv2.imshow('after skin', without_skin)
    cv2.waitKey(0)


skin_removal_test()