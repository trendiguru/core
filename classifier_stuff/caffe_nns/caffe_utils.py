# coding: utf-8
__author__ = 'jeremy'

# Run the script with anaconda-python
# $ /home/<path to anaconda directory>/anaconda/bin/python LmdbClassification.py
import sys
import numpy as np
import lmdb
import caffe
from collections import defaultdict
import socket

def conf_mat(deploy_prototxt_file_path,caffe_model_file_path,test_lmdb_path,meanB=128,meanG=128,meanR=128)
#    caffe.set_mode_gpu()

    # Modify the paths given below

    # Extract mean from the mean image file
#    mean_blobproto_new = caffe.proto.caffe_pb2.BlobProto()
  #  f = open(mean_file_binaryproto, 'rb')
   # mean_blobproto_new.ParseFromString(f.read())
   # mean_image = caffe.io.blobproto_to_array(mean_blobproto_new)
    #f.close()

    # CNN reconstruction and loading the trained weights
    net = caffe.Net(deploy_prototxt_file_path, caffe_model_file_path, caffe.TEST)

    count = 0
    correct = 0
    matrix = defaultdict(int) # (real,pred) -> int
    labels_set = set()

    lmdb_env = lmdb.open(test_lmdb_path)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()

    for key, value in lmdb_cursor:
        datum = caffe.proto.caffe_pb2.Datum()
            datum.ParseFromString(value)
            label = int(datum.label)
            image = caffe.io.datum_to_array(datum)
            image = image.astype(np.uint8)
#        out = net.forward_all(data=np.asarray([image]) - mean_image)
#        image[:,:,0] = image[:,:,0]- meanB
  #      image[:,:,1] = image[:,:,1]- meanB
    #    image[:,:,2] = image[:,:,2] - meanB
        thedata = np.asarray([image])
        out = net.forward_all(thedata)
        plabel = int(out['prob'][0].argmax(axis=0))
        count += 1
        iscorrect = label == plabel
        correct += (1 if iscorrect else 0)
        matrix[(label, plabel)] += 1
        labels_set.update([label, plabel])

        if not iscorrect:
                print("\rError: key = %s, expected %i but predicted %i" % (key, label, plabel))
            sys.stdout.write("\rAccuracy: %.1f%%" % (100.*correct/count))
            sys.stdout.flush()

    print("\n" + str(correct) + " out of " + str(count) + " were classified correctly")
    print ""
    print "Confusion matrix:"
    print "(r , p) | count"
    for l in labels_set:
        for pl in labels_set:
            print "(%i , %i) | %i" % (l, pl, matrix[(l,pl)])

def load_net(prototxt,caffemodel,mean_B=128,mean_G=128,mean_R=128,image='../../images/female1.jpg',image_width=128,image_height=128,image_depth=3,batch_size=256):
        host = socket.gethostname()
        print('host:'+str(host))
        pc = False
        caffe.set_mode_gpu()
        if host == 'jr-ThinkPad-X1-Carbon':
            pc = True
            caffe.set_mode_cpu()
        net = caffe.Net(prototxt,caffemodel,caffe.TEST)
        # see http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/00-classification.ipynb
    #    mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
     #   mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
        mu = [mean_B,mean_G,mean_R]
        print 'mean-subtracted values:', zip('BGR', mu)

        # create transformer for the input called 'data'
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

    #    transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
        transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
        transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
    #    transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
    # set the size of the input (we can skip this if we're happy
    #  with the default; we can also change it later, e.g., for different batch sizes)
        net.blobs['data'].reshape(batch_size,        # batch size
                                  image_depth,         # 3-channel (BGR) images
                                 image_width, image_height)  # image size is 227x227
        image = caffe.io.load_image(image)
        transformed_image = transformer.preprocess('data', image)
        fig = plt.figure()
        fig.savefig('out.png')
    #    plt.imshow(image)
    # copy the image data into the memory allocated for the net
        net.blobs['data'].data[...] = transformed_image

        ### perform classification
        output = net.forward()

        output_prob = output['prob'][0]  # the output probability vector for the first image in the batch

        print 'predicted classes:', output_prob
        print 'predicted class is:', output_prob.argmax()


def test_net(prototxt,caffemodel, db_path):
    net = caffe.Net(prototxt, caffemodel,caffe.TEST)
    caffe.set_mode_cpu()
    lmdb_env = lmdb.open(db_path)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    count = 0
    correct = 0
    max_to_test = 100
    for key, value in lmdb_cursor:
        print "Count:"
        print count
        count = count + 1
        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(value)
        label = int(datum.label)
        image = caffe.io.datum_to_array(datum)
        image = image.astype(np.uint8)
        print('image size'+len(image)+' shape:'+str(image.shape))
        out = net.forward_all(data=np.asarray([image]))
        predicted_labels = out['prob'][0]
        most_probable_label = out['prob'][0].argmax(axis=0)
        if label == most_probable_label[0][0]:
            correct = correct + 1
        print("Label is class " + str(label) + ", predicted class is " + str(most_probable_label[0][0]))
        if count == max_to_test:
            break
    print(str(correct) + " out of " + str(count) + " were classified correctly")

def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in xrange(0, image.shape[0], stepSize):
		for x in xrange(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def detect_with_scale_pyramid_and_sliding_window(img_arr,caffemodel):

host = socket.gethostname()
print('host:'+str(host))

if __name__ == "__main__":
    if host == "":
#        deploy_prototxt
#        conf_mat(deploy_prototxt_file_path,caffe_model_file_path,test_lmdb_path,meanB=128,meanG=128,meanR=128)


