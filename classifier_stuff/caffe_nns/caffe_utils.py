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
import imutils
import matplotlib as plt

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

def get_nn_answer(prototxt,caffemodel,mean_B=128,mean_G=128,mean_R=128,image_filename='../../images/female1.jpg',image_width=150,image_height=200):
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
        print 'mean-subtracted values:',  mu

        # create transformer for the input called 'data'
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

        transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
        transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
   #     transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
        transformer.set_raw_scale('data', 1.0/255)      # rescale from [0, 1] to [0, 255]
    #    transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
    # set the size of the input (we can skip this if we're happy
    #  with the default; we can also change it later, e.g., for different batch sizes)
 #       net.blobs['data'].reshape(batch_size,        # batch size
 #                                 image_depth,         # 3-channel (BGR) images
 #                                image_width, image_height)  # image size is 227x227
            #possibly use cv2.imread here instead as that's how i did it in lmdb_utils
        image = caffe.io.load_image(image_filename)
        cv2.imshow(image_filename,image)
#        fig = plt.figure()
#        fig.savefig('out.png')
        transformed_image = transformer.preprocess('data', image)
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
# loop over the image pyramid
    for resized in pyramid(img_arr, scale=1.5):
        # loop over the sliding window for each layer of the pyramid
        for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(winW, winH)):
            # if the window does not meet our desired window size, ignore it
            if window.shape[0] != winH or window.shape[1] != winW:
                continue

            # THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
            # MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
            # WINDOW

            # since we do not have a classifier, we'll just draw the window
            clone = resized.copy()
            cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
            cv2.imshow("Window", clone)
            cv2.waitKey(1)
            time.sleep(0.025)



def pyramid(image, scale=1.5, minSize=(30, 30)):
	# yield the original image
	yield image

	# keep looping over the pyramid
	while True:
		# compute the new dimensions of the image and resize it
		w = int(image.shape[1] / scale)
		image = imutils.resize(image, width=w)

		# if the resized image does not meet the supplied minimum
		# size, then stop constructing the pyramid
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break

		# yield the next image in the pyramid
		yield image

host = socket.gethostname()
print('host:'+str(host))

if __name__ == "__main__":

    if host == 'jr-ThinkPad-X1-Carbon':
        pass
    else:
        prototxt = 'home/jeremy/core/classifier_stuff/caffe_nns/alexnet10_binary_dresses/my_solver_test.prototxt'
        caffemodel = 'home/jeremy/core/classifier_stuff/caffe_nns/alexnet10_binary_dresses/net_iter_9000.caffemodel'

        img_filename = 'home/jeremy/core/classifier_stuff/caffe_nns/dataset/cropped/retrieval_dresses/product_11111_photo_65075.jpg'

    get_nn_answer(prototxt,caffemodel,mean_B=128,mean_G=128,mean_R=128,image_filename=img_filename,image_width=150,image_height=200):
#        deploy_prototxt
#        conf_mat(deploy_prototxt_file_path,caffe_model_file_path,test_lmdb_path,meanB=128,meanG=128,meanR=128)


