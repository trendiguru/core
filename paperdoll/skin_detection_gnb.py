# from /49491-227766-1-PB.pdf
# __author__ = 'jeremy'
import cv2

from sklearn.naive_bayes import GaussianNB


gnb = GaussianNB()
gnb.fit(data, target)
# Skin detection can be performed converting the input image [Figure 9 (c)] to the L* a* b*
# color space, and then reshaping and slicing in the same way as the training image. The
# predict method of GaussianNB performs the classification. The resulting classification
#vector can be reshaped to the original image dimensions for visualization [Figure 9 (d) and
#(e)]:

test_bgr = cv2.imread('../data/thiago.jpg')
test = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2LAB)
M_tst, N_tst, _ = test.shape
data = test.reshape(M_tst * N_tst, -1)[:, 1:]
skin_pred = gnb.predict(data)
S = skin_pred.reshape(M_tst, N_tst)