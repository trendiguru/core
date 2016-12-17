"""
stages:
0. create a new folder and copy all the caffemodels + protos to there
1. create a new prototxt with the same root layers but many output leafs
2. load that proto in train mode
3. load many nets in Test mode
4. create a dict where each key is a pointer to the new net params
5. assign the params from the old nets to the ndict by keys
"""

import caffe
import os
import argparse


def copy_net_params(params_new, params_base):
    for pr in params_base.keys():
        print('param key {} len new {} len base {}'.format(pr,params_new[pr].shape,params_base[pr].shape))
        assert(params_new[pr].shape==params_base[pr].shape)
#possibly:
        # params_new[pr] = params_base[pr]
#or even
        #params_new = params_base
        for i in range(len(params_new[pr])):
            print('param {} weightshape {}'.format(i,params_new[pr][i].data.shape,params_base[pr][i].data.shape))
            params_new[pr][i].data = params_base[pr][i].data
    return params_new

def copy_layer_params(dest_net,dest_layer,source_net,source_layer):
    assert(dest_net[dest_layer].shape==source_net[source_layer].shape)
    print('copying suorcce layer {} to dest layer {}'.format(source_layer,dest_layer))

    for i in range(len(source_net.params[source_layer])):
        print('dest layer {}[{}] shape {} source layer {}[{}] shape  {}'.format(dest_layer,i,dest_net[dest_layer][i].data.shape,source_layer,i,source_net[source_layer][i].data.shape))
        dest_net.params[dest_layer][0].data = source_net.params[source_layer][0].data

    return dest_layer

def get_user_input():
    parser = argparse.ArgumentParser(description='"@@@ Many2One @@@')
    parser.add_argument('-f', '--folder', dest="path2folder",
                        help='path to the folder containing the trained models', required=True)
    parser.add_argument('-p', '--prototxt', dest="protoname",
                        help='name of the deploy prototxt', required=True)
    parser.add_argument('-o', '--output', dest="modelname",
                        help='name of the new model', required=True)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    user_input = get_user_input()
    folder_path = '/'.join(['.', user_input.path2folder])
    folder_path = user_input.path2folder
    all_files_in_dir = os.listdir(folder_path)
    print all_files_in_dir
    assert os.path.isfile(os.path.join(folder_path,user_input.protoname)), 'new prototxt file {} not found!'.format(user_input.protoname)

    model_files = [f for f in all_files_in_dir if '.caffemodel' in f]
    if user_input.modelname in model_files:
        model_files.remove(user_input.modelname)
    print('modelfiles to add:'+str(model_files))
    proto_files = [f for f in all_files_in_dir if '.prototxt' in f and not 'solver' in f]
    assert len(proto_files)==2, 'base prototxt file is missing!'
    assert len(model_files)>=2, 'one or fewer model files found'
    proto_files.remove(user_input.protoname)
    print('protofiles to read:'+str(proto_files))
    # load new net
#    protopath = '/'.join([folder_path, user_input.protoname])
    protopath = user_input.protoname
#    modelpath = '/'.join([folder_path, model_files[0]])#
    modelpath = model_files[0]
    net_new = caffe.Net(protopath, caffe.TEST,weights=modelpath)
    print('loaded model {} defined by proto {}'.format(model_files[0],user_input.protoname))
    raw_input('ret to cont')
    modelpath = '/'.join([folder_path, proto_files[0]])
    modelpath = os.path.join(folder_path,proto_files[0])
    nets = (caffe.Net(os.path.join(folder_path,cfm), caffe.TEST,weights=modelpath) for cfm in model_files)
    print('loaded models {} defined by proto {}'.format(model_files,proto_files[0]))

    # weights_dict(net_new.params, nets.next().params)
    nets.next()
    for i in range(1, len(model_files)):
        net_orig = nets.next()
        lower_fully_connected = 2  #e.g. fc2_0 is the first(lowest) fully connected of net 0
        last_fully_connected = 5  #e.g. fc5_2 is the last fullyconnected of net2
        for j in range(lower_fully_connected, last_fully_connected):

            fc_orig = 'fc{}_0'.format(j)
            fc_new = 'fc{}_{}'.format(j, i)
            net_new.params = copy_layer_params(net_new,fc_new,net_ori,fc_orig)
#            net_new.params[tmp_fc_new][0].data.flat = tmp_net.params[tmp_fc_base][0].data.flat
#            if len(net_new.params[tmp_fc_new]) == 2:
 #               net_new.params[tmp_fc_new][1].data[...] = tmp_net.params[tmp_fc_base][1].data
#            if len(net_new.params[tmp_fc_new]) > 2:
#                print('uh o')

    net_new.save('/'.join([folder_path, user_input.modelname]))

    nets.close()
    del net_new

    print 'DONE!'



