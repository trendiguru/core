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
import cv2
import numpy as np

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


def copy_layer_params(dest_net_params,dest_layer,source_net_params,source_layer):
    assert(dest_net_params[dest_layer].shape==source_net_params[source_layer].shape)
    print('copying source layer {} to dest layer {}, shape {}'.format(source_layer,dest_layer,source_net_params[source_layer].shape))

    for i in range(len(source_net_params[source_layer])):
        print('dest layer {}[{}] shape {} source layer {}[{}] shape  {}'.format(dest_layer,i,dest_net_params[dest_layer][i].data.shape,source_layer,i,source_net_params[source_layer][i].data.shape))
        dest_net_params[dest_layer][i].data = source_net_params[source_layer][i].data

    return dest_net_params

def get_user_input():
    parser = argparse.ArgumentParser(description='"@@@ Many2One @@@')
    parser.add_argument('-f', '--folder', dest="path2folder",
                        help='path to the folder containing the trained models', required=True)
    parser.add_argument('-d', '--deployproto', dest="dest_proto",
                        help='name of the deploy (destination) prototxt', required=True)
    parser.add_argument('-s', '--sourceproto', dest="source_proto",
                        help='name of the source (train/test) prototxt', required=False)
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

def inspect_net(proto='ResNet-101-deploy.prototxt',caffemodel='three_heads.caffemodel'):
    net = caffe.Net(proto, caffe.TEST,weights=caffemodel)
    conv1 = net.params['conv1'][0].data
    print('conv1 params shape {} mean {} std {}'.format(conv1.shape,np.mean(conv1),np.std(conv1)))

    bn1_0 = net.params['bn_conv1'][0].data
    print('bn0 params shape {} mean {} std {}'.format(bn1_0.shape,np.mean(bn1_0),np.std(bn1_0)))
    bn1_1 = net.params['bn_conv1'][1].data
    print('bn1 params shape {} mean {} std {}'.format(bn1_1.shape,np.mean(bn1_1),np.std(bn1_1)))
    bn1_2 = net.params['bn_conv1'][2].data
    print('bn2 params shape {} mean {} std {}'.format(bn1_2.shape,np.mean(bn1_2),np.std(bn1_2)))

    scale1_0 = net.params['scale_conv1'][0].data
    print('scale0 params shape {} mean {} std {}'.format(scale1_0.shape,np.mean(scale1_0),np.std(scale1_0)))
    scale1_1 = net.params['scale_conv1'][1].data
    print('scale1 params shape {} mean {} std {}'.format(scale1_1.shape,np.mean(scale1_1),np.std(scale1_1)))

#    img_arr = cv2.imread('/usr/lib/python2.7/dist-packages/trendi/images/female1.jpg')
    img_arr = cv2.imread('/home/jeremy/projects/core/images/female1.jpg')
    img_arr=cv2.resize(img_arr,(224,224))
    print('in shape '+str(img_arr.shape))
    img_arr=np.transpose(img_arr,[2,0,1])
    img_arr = np.array(img_arr, dtype=np.float32)
    print('out shape '+str(img_arr.shape))
    net.blobs['data'].reshape(1,*img_arr.shape)
    net.blobs['data'].data[...] = img_arr
    net.forward()
    print('data mean {} std {}'.format(np.mean(net.blobs['data'].data),np.std(net.blobs['data'].data)))
    print('conv1 mean {} std {}'.format(np.mean(net.blobs['conv1'].data),np.std(net.blobs['conv1'].data)))
    print('pool1 mean {} std {}'.format(np.mean(net.blobs['pool1'].data),np.std(net.blobs['pool1'].data)))

    print('demeaned image now:')
    img_arr = img_arr - 112
    net.blobs['data'].data[...] = img_arr
    net.forward()
    print('data mean {} std {}'.format(np.mean(net.blobs['data'].data),np.std(net.blobs['data'].data)))
    print('conv1 mean {} std {}'.format(np.mean(net.blobs['conv1'].data),np.std(net.blobs['conv1'].data)))
    print('pool1 mean {} std {}'.format(np.mean(net.blobs['pool1'].data),np.std(net.blobs['pool1'].data)))

    print('scaled img')
    img_arr = img_arr / 256
    net.blobs['data'].data[...] = img_arr
    net.forward()
    print('data mean {} std {}'.format(np.mean(net.blobs['data'].data),np.std(net.blobs['data'].data)))
    print('conv1 mean {} std {}'.format(np.mean(net.blobs['conv1'].data),np.std(net.blobs['conv1'].data)))
    print('pool1 mean {} std {}'.format(np.mean(net.blobs['pool1'].data),np.std(net.blobs['pool1'].data)))

    conv1 = net.params['conv1'][0].data
    print('conv1 params shape {} mean {} std {}'.format(conv1.shape,np.mean(conv1),np.std(conv1)))

    bn1_0 = net.params['bn_conv1'][0].data
    print('bn0 params shape {} mean {} std {}'.format(bn1_0.shape,np.mean(bn1_0),np.std(bn1_0)))
    bn1_1 = net.params['bn_conv1'][1].data
    print('bn1 params shape {} mean {} std {}'.format(bn1_1.shape,np.mean(bn1_1),np.std(bn1_1)))
    bn1_2 = net.params['bn_conv1'][2].data
    print('bn2 params shape {} mean {} std {}'.format(bn1_2.shape,np.mean(bn1_2),np.std(bn1_2)))


if __name__ == "__main__":
    user_input = get_user_input()
    folder_path = user_input.path2folder
    all_files_in_dir = os.listdir(folder_path)
    proto_files = [f for f in all_files_in_dir if '.prototxt' in f and not 'solver' in f]
    if user_input.source_proto is not None:
        source_proto = user_input.source_proto
    else:
    #till now the dest proto is a superset of the source so its ok to use dest as source, it will
    #have extra outputs and layers that will just get ignored
        source_proto = user_input.dest_proto
    dest_proto = user_input.dest_proto
    assert os.path.isfile(os.path.join(folder_path,source_proto)), 'source prototxt file {} not found!'.format(source_proto)
    assert os.path.isfile(os.path.join(folder_path,dest_proto)), 'dest prototxt file {} not found!'.format(dest_proto)

    model_files = [f for f in all_files_in_dir if '.caffemodel' in f]
    if user_input.modelname in model_files:
        model_files.remove(user_input.modelname)
    assert len(model_files)>=1, 'no extra model files found '
    first_model_path = os.path.join(folder_path,model_files[0])
    print('initial model:'+str(first_model_path))
    print('modelfiles to add:'+str(model_files[1:]))
    raw_input('loading net {} using proto {} (ret to cont)'.format(first_model_path,dest_proto))

    caffe.set_mode_cpu()
     destination_net = caffe.Net(dest_proto, caffe.TEST,weights=first_model_path)
    print('loaded model {} defined by proto {}'.format(first_model_path,dest_proto))
#    modelpath = '/'.join([folder_path, proto_files[0]])

    nets = []
    for i in range(len((model_files))):
        cfm_base = model_files[i+1] #first model is used as base, 2nd and subsequent added to it
        caffemodel = os.path.join(folder_path,cfm_base)
        prototxt = os.path.join(folder_path,source_proto)
        raw_input('adding net {} using proto {} (ret to cont)'.format(caffemodel,prototxt))
        net = caffe.Net(prototxt, caffe.TEST,weights=caffemodel)
        nets.append(net)
    print('loaded {} models {}\ndefined by proto {}'.format(len(model_files),model_files,prototxt))

    # weights_dict(net_new.params, nets.next().params)
#    nets.next()
   #add final layers from nets to destination net
    for i in range(len(model_files)):
        net_orig = nets[i]
        lower_fully_connected = 2  #e.g. fc2_0 is the first(lowest) fully connected of net 0, fc2_2 is first of net 2
        upper_fully_connected = 4  #e.g. fc4_0 is the last fullyconnected of net0, fc4_2 is last of net2
        destination_output = i+1
        for j in range(lower_fully_connected, upper_fully_connected):
            fc_orig = 'fc{}_0'.format(j)
            fc_dest = 'fc{}_{}'.format(j, destination_output)
            destination_net.params = copy_layer_params(destination_net.params,fc_dest,net_orig.params,fc_orig)
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



