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


def weights_dict(params_new, params_base):
    for pr in params_base.keys():
        params_new[pr][0].data.flat = params_base[pr][0].data.flat
        print pr
        if len(params_new[pr]) == 2:
            params_new[pr][1].data[...] = params_base[pr][1].data


def get_user_input():
    parser = argparse.ArgumentParser(description='"@@@ Many2One @@@')
    parser.add_argument('-f', '--folder', dest="path2folder",
                        help='path to the folder containing the trained models', required=True)
    parser.add_argument('-p', '--prototxt', dest="protoname",
                        help='name of the new prototxt', required=True)
    parser.add_argument('-o', '--output', dest="modelname",
                        help='name of the new model', required=True)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    user_input = get_user_input()
    folder_path = '/'.join(['.', user_input.path2folder])
    all_files_in_dir = os.listdir(folder_path)
    assert user_input.protoname in all_files_in_dir, 'new prototxt file not in folder!'

    model_files = [f for f in all_files_in_dir if '.caffemodel' in f]
    if user_input.modelname in model_files:
        model_files.remove(user_input.modelname)
    proto_files = [f for f in all_files_in_dir if '.prototxt' in f]
    assert len(proto_files)==2, 'base prototxt file is missing!'
    proto_files.remove(user_input.protoname)
    # load new net
    net_new = caffe.Net('/'.join([folder_path, user_input.protoname]),'/'.join([folder_path, model_files[0]]), caffe.TEST)
    nets = (caffe.Net('/'.join([folder_path, proto_files[0]]),'/'.join([folder_path, cfm]), caffe.TEST) for cfm in model_files)

    # weights_dict(net_new.params, nets.next().params)
    nets.next()
    for i in range(1, len(model_files)):
        tmp_net = nets.next()
        for j in range(2, 5):
            tmp_fc_base = 'fc{}_0'.format(j)
            tmp_fc_new = 'fc{}_{}'.format(j, i)
            net_new.params[tmp_fc_new][0].data.flat = tmp_net.params[tmp_fc_base][0].data.flat
            if len(net_new.params[tmp_fc_new]) == 2:
                net_new.params[tmp_fc_new][1].data[...] = tmp_net.params[tmp_fc_base][1].data
    net_new.save('/'.join([folder_path, user_input.modelname]))

    nets.close()
    del net_new

    print 'DONE!'



