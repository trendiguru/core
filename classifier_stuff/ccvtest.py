import sys
from PIL import Image
from ccv import PY_CCV_IO_GRAY, DenseMatrix, ClassifierCascade, detect_objects
import cv2
import ccv

matrix = DenseMatrix()

cv2_im = cv2.imread(sys.argv[1])
#gray_image = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2GRAY)

pil_im = Image.fromarray(cv2.cvtColor(cv2_im,cv2.COLOR_BGR2RGB))
#pil_im = Image.fromarray(cv2_im)

#pil_im = Image.open(sys.argv[1])
st=bytearray(pil_im.tostring())
print('len:{0} w {1} h {2} mode {3}'.format(str(len(st)),str(pil_im.size[0]),str(pil_im.size[1]),str(pil_im.mode)))
#matrix.set_buf(st, pil_im.mode, pil_im.size[0], pil_im.size[1], PY_CCV_IO_GRAY)
matrix.set_buf(st, pil_im.mode, pil_im.size[0], pil_im.size[1])
print matrix
a=matrix.first_pixel()
print('first pixel'+str(a)) 
#print matrix._matrix
#        ccv_dense_matrix_t* image = 0;
# 6        ccv_read(argv[1], &image, CCV_IO_GRAY | CCV_IO_ANY_FILE);
# 7        ccv_write(image, argv[2], 0, CCV_IO_PNG_FILE, 0);
#ccv.ccv_write(pil_im,"testout.png", 0, CCV_IO_PNG_FILE, 0);

#matrix.set_file(sys.argv[1], PY_CCV_IO_GRAY)
cascade = ClassifierCascade()
cascade.read(sys.argv[2])

print detect_objects(matrix, cascade, 1)
