"""
stages:
1. create a new folder and copy all the caffemodels + protos to there
    - there should by equal number of protos and models - 1 and 1 under each name
2. create a new prototxt with the same root layers but many output leafs
3. load that proto in test mode with the first layer weights
4. load many nets in Test mode and fill in their weights
5. save new net to output folder

usage -
python   /usr/lib/python2.7/dist-packages/trendi/classifier_stuff/caffe_nns/make_hydra_from_caffemodels.py -f /data/jeremy/caffenets/hydra/production/ -d ResNet-152-deploy.prototxt -s ResNet-152-deploy.prototxt
ie deploy and source protos identical ; source can prob be deprecated to use specific source proto for each caffemodel (then the model final
layers can be different

"""

import caffe
from os import listdir, getcwd, chdir, mkdir
from os.path import splitext
from os.path import join as path_join
import argparse
import glob
import re
import json
import os

from trendi.classifier_stuff.caffe_nns import jrinfer
import pdb
import numpy as np
def test_hydra(proto='ResNet-101-deploy.prototxt',caffemodel='three_heads.caffemodel',gpu=0):
    #pants, shirt, dress
    urls = ['http://g04.a.alicdn.com/kf/HTB1BdwqHVXXXXcJXFXXq6xXFXXXz/2015-Fashion-Spring-Summer-Pants-Women-Straight-Career-Trousers-for-Office-Ladies-Black-Green-Pantalones-Women.jpg',
            'http://getabhi.com/image/cache/catalog/BARCODE:%20324BNZ61RBLUE/2-800x800.jpg',
            'http://myntra.myntassets.com/images/style/properties/Belle-Fille-Black-Maxi-Dress_e3e65039ce204cefb7590fc8ec10f1e9_images.jpg']
    backpack='http://blogs.cornell.edu/sarahl/files/2014/08/Backpacks-2iot4za.jpg'
    blazer = 'http://media.brostrick.com/wp-content/uploads/2015/02/24223725/penguin-stretch-seersucker-sports-jacket-blue-for-men-2016.jpg'
    cardigan = 'https://s-media-cache-ak0.pinimg.com/originals/4c/a0/5b/4ca05ba61e6d33b51a6f90ccc290d0da.jpg'
    jrinfer.infer_many_hydra([backpack,blazer,cardigan],proto,caffemodel,out_dir='./',gpu=gpu)

def mega_test_hydra(proto='/data/jeremy/caffenets/hydra/production/output/hydra_out.prototxt',
                    caffemodel='/data/jeremy/caffenets/hydra/production/output/hydra_out.prototxt',gpu=0):

    dirs = ['/data/jeremy/image_dbs/tg/google/backpack/kept',
            '/data/jeremy/image_dbs/tg/google/hat/kept',
            '/data/jeremy/image_dbs/deep_fashion/category_and_attribute_prediction/img_256x256/Quilted_Bomber_Jacket']
    correct_indices = [0,6,7] #backpack, hat, jacket
    for dir,correct_index in zip(dirs,correct_indices):

        backpack='http://blogs.cornell.edu/sarahl/files/2014/08/Backpacks-2iot4za.jpg'
        blazer = 'http://media.brostrick.com/wp-content/uploads/2015/02/24223725/penguin-stretch-seersucker-sports-jacket-blue-for-men-2016.jpg'
        cardigan = 'https://s-media-cache-ak0.pinimg.com/originals/4c/a0/5b/4ca05ba61e6d33b51a6f90ccc290d0da.jpg'
        images = [os.path.join(dir,f) for f in dir if '.jpg' in f]
        images=[backpack,blazer,cardigan]
        answers = jrinfer.infer_many_hydra(images,proto,caffemodel,out_dir='./',gpu=gpu)

        n_true_pos = 0
        n_false_neg = 0
        for answer in answers:
            output_of_interest = answer[correct_index]
            if answer[1]> answer[0]:
                n_true_pos += 1
            else:
                n_false_neg += 1
        print('true pos {} false neg {} approx.acc {}'.format(n_true_pos,n_false_neg,float(n_true_pos/(n_true_pos+n_false_neg))))
#
def show_all_params(proto,caffemodel,filter='',gpu=0):
    '''
    print all params
    '''
    print('starting show_all_params')
  #  pdb.set_trace()
    caffe.set_mode_gpu()
    caffe.set_device(gpu)

    net = caffe.Net(proto, caffe.TEST,weights=caffemodel)
    print('starting show_all_params')
    all_params = [p for p in net.params if filter in p]
    print('starting show_all_params')

    print('showing params ')
    print('all params in net1:'+str(all_params))
    layers_to_compare = all_params
    for layer in layers_to_compare:
        for i in range(len(net.params[layer])):
            params = net.params[layer][i].data
            print('net {}[{}] params shape {} mean {} std {}'.format(layer,i,params.shape,np.mean(params),np.std(params)))

def show_all_layers(proto,caffemodel,filter='',gpu=None):
    '''
    print all params
    '''

    if(gpu):
        caffe.set_mode_gpu()
        caffe.set_device(gpu)
    else:
        caffe.set_mode_cpu()
        
    net = caffe.Net(proto, caffe.TEST,weights=caffemodel)
    all_params = [p for p in net.layers if filter in p]

    print('showing params ')
    print('all params in net1:'+str(all_params))
    layers_to_compare = all_params
    for layer in layers_to_compare:
        for i in range(len(net.params[layer])):
            params = net.params[layer][i].data
            print('net {}[{}] params shape {} mean {} std {}'.format(layer,i,params.shape,np.mean(params),np.std(params)))


def create_new_proto(names, lastlayer, output_name):
    #todo - put model name into the estimate layer name so output is unambiguous
    '''
    take a set of protxts, copy 1st entirely into new_proto, then copy all layers past last common layer
    'lastlayer' into new_proto
    :param names:
    :param lastlayer:
    :param output_name:
    :return:
    '''
    new_proto = open(output_name, 'w')

    for i in range(len(names)):#
#        f = open('{}.prototxt'.format(names[i]), 'r')
        f = open(names[i], 'r')
        print('getting layers from layer {} {}'.format(i,names[i]))
        if i == 0:
            print('copying entire net')
            for line in f:
                if len(line) == 1:
                    continue
                new_proto.write('{}'.format(line))
        else:
            lastlayer_flag = False
            start_flag = False
            for line in f:
                if 'name' in line and lastlayer in line:
                    lastlayer_flag = True
                    print('line with name and lastlayer:'+str(line))
                if lastlayer_flag and 'layer' in line:
                    start_flag = True
                if not start_flag or len(line) == 1:
                    continue
                if lastlayer in line:
                    print('keeping line asis'+str(line))
                    current_layer_is_lastlayer = False
                elif any(n in line for n in ['name', 'bottom', 'top']) and not lastlayer in line:
                    line_parts = re.split('"', line)
                    arg = '__'.join([line_parts[1], str(i)]) #add __n to layer name
                    line_parts[1] = '"'+arg+'"'   #need to keep layer name in quotes in prototxt
                    line = ''.join(line_parts)
                print('adding new line :'+str(line))
                new_proto.write('{}'.format(line))
        f.close()
    new_proto.close()
    print 'prototxt ready'


def verify_protos_vs_models(protos, models):
    # check if there is equal number of files
    assert len(proto_files) == len(model_files), \
        'there is suppose to be equal number of protos and caffemodels!'
    # check that each file name as both prototxt & caffemodel
    names = {i: splitext(x)[0] for i, x in enumerate(protos)}
    assert all('{}.caffemodel'.format(names[n]) in models for n in names), \
        'not all nets have both prototxt and caffemodel files'

    return names


def get_user_input():
    parser = argparse.ArgumentParser(description='"@@@ Hydra-Maker @@@')
    parser.add_argument('-f', '--folder', dest="path2folder",
                        help='path to the folder containing the trained models', required=True)
    parser.add_argument('-l', '--lastlayer', dest="lastCommonLayer",
                        help='name of the last common layer', required=True)
    parser.add_argument('-o', '--output', dest="newName",
                        help='name of the new model', required=True)
    parser.add_argument('-g', '--gpu', dest="gpu", default=0,
                        help='name of the new model', required=False)
    args = parser.parse_args()
    args.gpu = int(args.gpu)
    return args

if __name__ == "__main__":
    user_input = get_user_input()
    folder_path = path_join(getcwd(), user_input.path2folder)
    chdir(folder_path)

    if 'output/' not in glob.glob('output/'):
        mkdir('output', 0755)

    output_folder = '/'.join([getcwd(), 'output'])
    all_files_in_dir = listdir(folder_path)
    model_files = [f for f in glob.glob('*.caffemodel')]
    model_files.sort()
    print('models:'+str(model_files))
    proto_files = [f for f in glob.glob('*.prototxt') if f.replace('prototxt','caffemodel') in model_files]
    proto_files.sort()
    print('protos:'+str(proto_files))


    nets_names = verify_protos_vs_models(proto_files, model_files)
    print('netnames '+str(nets_names))
    new_prototxt_path = '{}/{}.prototxt'.format(output_folder, user_input.newName)
    print('new prototxt '+str(new_prototxt_path))

    #this proto has all the  layers of all nets specified
    create_new_proto(proto_files, user_input.lastCommonLayer, new_prototxt_path)

    first_model_path = ''.join([nets_names[0], '.caffemodel'])
    nets_names.pop(0)
    caffe.set_mode_gpu()
    print('gpu:'+str(user_input.gpu))
    caffe.set_device(user_input.gpu)
    print('getting new net')
    net_new = caffe.Net(new_prototxt_path, caffe.TEST, weights=first_model_path)
    raw_input('got new net, return to continue')
    net_new_layers = net_new.params.keys()
    #get old model values into new caffemodel
    print('loading old values into new model')

    #get first net info
    net_tmp = caffe.Net(proto_files[0], caffe.TEST, weights=first_model_path)
    first_net_fc_layers = [l for l in net_tmp.params if 'fc' in l]
    net_info = [[proto_files[0],first_model_path,first_net_fc_layers]]
    del net_tmp

    for k in nets_names:
        cfm = ''.join([nets_names[k], '.caffemodel'])
        prt = ''.join([nets_names[k], '.prototxt'])
        print('loading k: {} model {} and proto {} '.format(k,cfm,prt))
        net_tmp = caffe.Net(prt, caffe.TEST, weights=cfm)
        print('loaded k: {}  model {} and proto {} '.format(k,cfm,prt))
        params_to_replace = [p for p in net_new_layers if p.endswith('__{}'.format(str(k)))]
        print('params to replace {}'.format(params_to_replace))
        net_info.append([cfm,prt,params_to_replace])
  #      raw_input('return to continue')
        for pr in params_to_replace:
            pr_tmp =  pr.split('__')[0]  #get layername part of layername__x

            print('copying values from {} to {} '.format(pr_tmp,pr))
            for i in range(len(net_new.params[pr])):
                net_new.params[pr][i].data[...] = net_tmp.params[pr_tmp][i].data
#                net_new.params[pr][i].data = net_tmp.params[pr_tmp][i].data
#            dest_net_params[dest_layer][i].data[...] = source_net_params[source_layer][i].data


        del net_tmp

    net_new_path = '{}/{}.caffemodel'.format(output_folder, user_input.newName)
    net_new.save(net_new_path)

    del net_new
    print 'DONE!'

    net_info_filename = os.path.join(output_folder,user_input.newName+'_netinfo.txt')
    with open(net_info_filename,'w') as fp:
        json.dump(net_info,fp,indent = 4)


