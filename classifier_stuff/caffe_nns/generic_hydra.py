"""
stages:
1. create a new folder and copy all the caffemodels + protos to there
    - there should by equal number of protos and models - 1 and 1 under each name

2. create a new prototxt with the same root layers but many output leafs
3. load that proto in test mode with the first layer weights
4. load many nets in Test mode and fill in their weights
5. save new net to output folder
"""

import caffe
from os import listdir, getcwd, chdir, mkdir
from os.path import splitext
from os.path import join as path_join
import argparse
import glob
import re

from trendi.classifier_stuff.caffe_nns import jrinfer

# def copy_net_params(params_new, params_base):
#     for pr in params_base.keys():
#         print('param key {} len new {} len base {}'.format(pr,params_new[pr].shape,params_base[pr].shape))
#         assert(params_new[pr].shape==params_base[pr].shape)
# #possibly:
#         # params_new[pr] = params_base[pr]
# #or even
#         #params_new = params_base
#         for i in range(len(params_new[pr])):
#             print('param {} weightshape {}'.format(i,params_new[pr][i].data.shape,params_base[pr][i].data.shape))
#             params_new[pr][i].data = params_base[pr][i].data
#     return params_new
#
# def copy_layer_params(dest_net,dest_layer,source_net,source_layer):
#     assert(dest_net[dest_layer].shape==source_net[source_layer].shape)
#     print('copying suorcce layer {} to dest layer {}'.format(source_layer,dest_layer))
#
#     for i in range(len(source_net.params[source_layer])):
#         print('dest layer {}[{}] shape {} source layer {}[{}] shape  {}'.format(dest_layer,i,dest_net[dest_layer][i].data.shape,source_layer,i,source_net[source_layer][i].data.shape))
#         dest_net.params[dest_layer][0].data = source_net.params[source_layer][0].data
#
#     return dest_layer



def test_hydra(proto='ResNet-101-deploy.prototxt',caffemodel='three_heads.caffemodel'):
    #pants, shirt, dress
    urls = ['http://g04.a.alicdn.com/kf/HTB1BdwqHVXXXXcJXFXXq6xXFXXXz/2015-Fashion-Spring-Summer-Pants-Women-Straight-Career-Trousers-for-Office-Ladies-Black-Green-Pantalones-Women.jpg',
            'http://getabhi.com/image/cache/catalog/BARCODE:%20324BNZ61RBLUE/2-800x800.jpg',
            'http://myntra.myntassets.com/images/style/properties/Belle-Fille-Black-Maxi-Dress_e3e65039ce204cefb7590fc8ec10f1e9_images.jpg']
    jrinfer.infer_many_hydra(urls,proto,caffemodel,out_dir='./',dims=(224,224),output_layers=['fc4_0','fc4_1','fc4_2'])


def create_new_proto(names, lastlayer, output_name):
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
        print('getting layers from {}'.format(names[i]))
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
    print('models:'+str(model_files))
    proto_files = [f for f in glob.glob('*.prototxt') if f.replace('prototxt','caffemodel') in model_files]
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
    for k in nets_names:
        cfm = ''.join([nets_names[k], '.caffemodel'])
        prt = ''.join([nets_names[k], '.prototxt'])
        net_tmp = caffe.Net(prt, caffe.TEST, weights=cfm)
        params_to_replace = [p for p in net_new_layers if p.endswith('__{}'.format(str(k)))]
        for pr in params_to_replace:
            pr_tmp = pr[:-3]
            for i in range(len(net_new.params[pr])):
                net_new.params[pr][i].data[...] = net_tmp.params[pr_tmp][i].data
#                net_new.params[pr][i].data = net_tmp.params[pr_tmp][i].data

#            dest_net_params[dest_layer][i].data[...] = source_net_params[source_layer][i].data



        del net_tmp

    net_new_path = '{}/{}.caffemodel'.format(output_folder, user_input.newName)
    net_new.save(net_new_path)

    del net_new
    print 'DONE!'



