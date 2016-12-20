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

from trendi.classifier_stuff.caffe_nns import jrinfer

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
    parser.add_argument('-d', '--destproto', dest="dest_proto",
                        help='name of the destination deploy prototxt', required=True)
    parser.add_argument('-s', '--sourceproto', dest="source_proto",
                        help='name of the source (train/test prototxt', required=False)
    parser.add_argument('-o', '--output', dest="modelname",
                        help='name of the new model', required=True)
    args = parser.parse_args()
    return args

def test_hydra(proto='ResNet-101-deploy.prototxt',caffemodel='three_heads.caffemodel'):
    #pants, shirt, dress
    urls = ['http://g04.a.alicdn.com/kf/HTB1BdwqHVXXXXcJXFXXq6xXFXXXz/2015-Fashion-Spring-Summer-Pants-Women-Straight-Career-Trousers-for-Office-Ladies-Black-Green-Pantalones-Women.jpg',
            'http://getabhi.com/image/cache/catalog/BARCODE:%20324BNZ61RBLUE/2-800x800.jpg',
            'http://myntra.myntassets.com/images/style/properties/Belle-Fille-Black-Maxi-Dress_e3e65039ce204cefb7590fc8ec10f1e9_images.jpg']

 #   for url in urls:
 #       jrinfer.infer_one_hydra(url,proto,caffemodel,out_dir='./',dims=(224,224),output_layers=['fc4_0','fc4_1','fc4_2'])
    jrinfer.infer_many_hydra(urls,proto,caffemodel,out_dir='./',dims=(224,224),output_layers=['fc4_0','fc4_1','fc4_2'])


if __name__ == "__main__":
    user_input = get_user_input()
    folder_path = '/'.join(['.', user_input.path2folder])
    folder_path = user_input.path2folder
    all_files_in_dir = os.listdir(folder_path)
    proto_files = [f for f in all_files_in_dir if '.prototxt' in f and not 'solver' in f]
    if user_input.source_proto is not None:
        source_proto = user_input.source_proto
    else:
        source_proto = user_input.dest_proto
    dest_proto = user_input.dest_proto
    assert os.path.isfile(os.path.join(folder_path,source_proto)), 'source prototxt file {} not found!'.format(source_proto)
    assert os.path.isfile(os.path.join(folder_path,dest_proto)), 'dest prototxt file {} not found!'.format(dest_proto)

    model_files = [f for f in all_files_in_dir if '.caffemodel' in f]
    model_files.remove(user_input.modelname)
    first_model_path = model_files[0]
    model_files.remove(first_model_path)
    print('initial model:'+str(first_model_path))
    print('modelfiles to add:'+str(model_files))
    raw_input('loading net {} using proto {} (ret to cont)'.format(first_model_path,dest_proto))
    assert len(model_files)>=1, 'no extra model files found '
    net_new = caffe.Net(dest_proto, caffe.TEST,weights=first_model_path)
    print('loaded model {} defined by proto {}'.format(first_model_path,dest_proto))
#    modelpath = '/'.join([folder_path, proto_files[0]])
    nets = []
    for i in range(len((model_files))):
        cfm_base = model_files[i]
 #       if 0*len(proto_files)==len(model_files):
 #           proto = proto_files[i]
  #      else:
        proto_base = source_proto
        caffemodel = os.path.join(folder_path,cfm_base)
        prototxt = os.path.join(folder_path,proto_base)
        raw_input('adding net {} using proto {} (ret to cont)'.format(caffemodel,prototxt))
        net = caffe.Net(prototxt, caffe.TEST,weights=caffemodel)
        nets.append(net)
    print('loaded models {} defined by proto {}'.format(model_files,prototxt))

    # weights_dict(net_new.params, nets.next().params)
#    nets.next()
    for i in range(1, len(model_files)):
        net_orig = nets[i]
        lower_fully_connected = 2  #e.g. fc2_0 is the first(lowest) fully connected of net 0
        last_fully_connected = 5  #e.g. fc5_2 is the last fullyconnected of net2
        for j in range(lower_fully_connected, last_fully_connected):
            fc_orig = 'fc{}_0'.format(j)
            fc_new = 'fc{}_{}'.format(j, i)
            #the below fails due to 'type net doesnt have expected attribute '__get_item'
#            net_new.params = copy_layer_params(net_new.params,fc_new.params,net_orig,fc_orig)
 #           assert(net_new[fc_new].shape==net_orig[fc_orig].shape)
#            assert(net_new[fc_new][0].data.shape==net_orig[fc_orig][0].data.shape)

#            print('copying source layer {} to dest layer {}'.format(fc_new,fc_orig))

#            for layer_level in range(len(net_new.params[fc_new])):#
#                print('orig layer {}[{}] shape {} new layer {}[{}] shape  {}'.format(fc_orig,layer_level,net_orig[fc_orig][layer_level].data.shape,fc_new,layer_level,net_new[fc_new][layer_level].data.shape))
#                net_new.params[fc_new][layer_level].data = net_orig.params[fc_orig][layer_level].data

            # net_new.params[tmp_fc_new][0].data.flat = tmp_net.params[tmp_fc_base][0].data.flat
            # if len(net_new.params[tmp_fc_new]) == 2:
            #     net_new.params[tmp_fc_new][1].data[...] = tmp_net.params[tmp_fc_base][1].data
            # if len(net_new.params[tmp_fc_new]) > 2:
            #     print('uh o')

    net_new.save('/'.join([folder_path, user_input.modelname]))

#    nets.close()
    del net_new

    print 'DONE!'



