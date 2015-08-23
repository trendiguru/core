__author__ = 'jeremy'

import warnings
import os
import collections
import numpy as np
import re

import cv2


CASCADE = "face.xml"
SAMPLES_DIREC = "samples"
LAUNCH_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__)))
CASCADE_PATH = os.path.join(LAUNCH_PATH, SAMPLES_DIREC, CASCADE)


class FaceRecognizer():
    def __init__(self):
        """
        Create a Face Recognizer Class using Fisher Face Recognizer. Uses
        OpenCV's FaceRecognizer class. Currently supports Fisher Faces.
        """
        self.supported = True
        self.model = None
        self.train_imgs = None
        self.train_labels = None
        self.csvfiles = []
        self.imageSize = None
        self.labels_dict = {}
        self.labels_set = []
        self.int_labels = []
        self.labels_dict_rev = {}
        if not hasattr(cv2, 'createFisherFaceRecognizer'):
            self.supported = False
            warnings.warn("Returning None. OpenCV >= 2.4.4 required.")
            return
        self.model = cv2.createFisherFaceRecognizer()

        # Not yet supported
        # self.eigenValues = None
        # self.eigenVectors = None
        # self.mean = None


    def train(self, images=None, labels=None, csvfile=None, delimiter=";"):
        """
        **SUMMARY**
        Train the face recognizer with images and labels.
        **PARAMETERS**
        * *images*    - A list of Images or ImageSet. All the images must be of
                        same size.
        * *labels*    - A list of labels(int) corresponding to the image in
                        images.
                        There must be at least two different labels.
        * *csvfile*   - You can also provide a csv file with image filenames
                        and labels instead of providing labels and images
                        separately.
        * *delimiter* - The delimiter used in csv files.
        **RETURNS**
        Nothing. None.
        **EXAMPLES**
        >>> f = FaceRecognizer()
        >>> imgs1 = ImageSet(path/to/images_of_type1)
        >>> labels1 = LabelSet("type1", imgs1)
        >>> imgs2 = ImageSet(path/to/images_of_type2)
        >>> labels2 = LabelSet("type2", imgs2)
        >>> imgs3 = ImageSet(path/to/images_of_type3)
        >>> labels3 = LabelSet("type3", imgs3)
        >>> imgs = concatenate(imgs1, imgs2, imgs3)
        >>> labels = concatenate(labels1, labels2, labels3)
        >>> f.train(imgs, labels)
        >>> imgs = ImageSet("path/to/testing_images")
        >>> print f.predict(imgs)
        Save Fisher Training Data
        >>> f.save("trainingdata.xml")
        Load Fisher Training Data and directly use without trainging
        >>> f1 = FaceRecognizer()
        >>> f1.load("trainingdata.xml")
        >>> imgs = ImageSet("path/to/testing_images")
        >>> print f1.predict(imgs)
        Use CSV files for training
        >>> f = FaceRecognizer()
        >>> f.train(csvfile="CSV_file_name", delimiter=";")
        >>> imgs = ImageSet("path/to/testing_images")
        >>> print f.predict(imgs)
        """

        if csvfile:
            images = []
            labels = []
            import csv

            try:
                f = open(csvfile, "rb")
            except IOError:
                warnings.warn("No such file found. Training not initiated")
                return None

            self.csvfiles.append(csvfile)
            filereader = csv.reader(f, delimiter=delimiter)
            for row in filereader:
                # images.append(Image(row[0]))
                print('reading {0} class {1}'.format(row[0], row[1]))
                img_arr = cv2.imread(row[0])
                if img_arr is None:
                    print('unsuccesful read of ' + str(row[0]))
                    continue
                images.append(img_arr)
                labels.append(row[1])

        print('labels:' + str(labels))
        if isinstance(labels, type(None)):
            warnings.warn("Labels not provided. Training not inititated.")
            return None

        self.labels_set = list(set(labels))
        i = 0
        for label in self.labels_set:
            self.labels_dict.update({label: i})
            self.labels_dict_rev.update({i: label})
            i += 1

        print('labelsdict:' + str(self.labels_dict))
        print('labelsdictrev:' + str(self.labels_dict_rev))
        if len(self.labels_set) < 2:
            warnings.warn("At least two classes/labels are required"
                          "for training. Training not inititated.")
            return None

        if len(images) != len(labels):
            warnings.warn("Mismatch in number of labels and number of"
                          "training images. Training not initiated.")
            return None

        self.imageSize = images[0].shape[:2]
        h, w = self.imageSize
        images = [img if img.shape[:2] == self.imageSize
                  else cv2.resize(img, (w, h)) for img in images]

        self.int_labels = [self.labels_dict[key] for key in labels]
        print('intlabels:' + str(self.int_labels))
        self.train_labels = labels
        print('trainlabels:' + str(self.train_labels))
        labels = np.array(self.int_labels)
        print('labels:' + str(labels))
        self.train_imgs = images
        # cv2imgs = [cv2.cvtColor(img, cv2.cv.CV_BGR2GRAY) for img in images]
        cv2imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]

        self.model.train(cv2imgs, labels)
        # Not yet supported
        # self.eigenValues = self.model.getMat("eigenValues")
        # self.eigenVectors = self.model.getMat("eigenVectors")
        # self.mean = self.model.getMat("mean")

    def predict(self, imgs):
        """
        **SUMMARY**
        Predict the class of the image using trained face recognizer.
        **PARAMETERS**
        * *image*    -  Image.The images must be of the same size as provided
                        in training.
        **RETURNS**
        * *label* - Class of the image which it belongs to.
        **EXAMPLES**
        >>> f = FaceRecognizer()
        >>> imgs1 = ImageSet(path/to/images_of_type1)
        >>> labels1 = LabelSet("type1", imgs1)
        >>> imgs2 = ImageSet(path/to/images_of_type2)
        >>> labels2 = LabelSet("type2", imgs2)
        >>> imgs3 = ImageSet(path/to/images_of_type3)
        >>> labels3 = LabelSet("type3", imgs3)
        >>> imgs = concatenate(imgs1, imgs2, imgs3)
        >>> labels = concatenate(labels1, labels2, labels3)
        >>> f.train(imgs, labels)
        >>> imgs = ImageSet("path/to/testing_images")
        >>> print f.predict(imgs)
        Save Fisher Training Data
        >>> f.save("trainingdata.xml")
        Load Fisher Training Data and directly use without trainging
        >>> f1 = FaceRecognizer()
        >>> f1.load("trainingdata.xml")
        >>> imgs = ImageSet("path/to/testing_images")
        >>> print f1.predict(imgs)
        Use CSV files for training
        >>> f = FaceRecognizer()
        >>> f.train(csvfile="CSV_file_name", delimiter=";")
        >>> imgs = ImageSet("path/to/testing_images")
        >>> print f.predict(imgs)
        """
        if not self.supported:
            warnings.warn("Fisher Recognizer is supported by OpenCV >= 2.4.4")
            return None
        h, w = self.imageSize
        images = [img if img.shape[:2] == self.imageSize
                  else cv2.resize(img, (w, h)) for img in imgs]

        if isinstance(imgs, np.ndarray):
            if imgs.shape[:2] != self.imageSize:
                image = cv2.resize(imgs, (w, h))
            cv2img = cv2.cvtColor(image, cv2.cv.CV_BGR2GRAY)
            label, confidence = self.model.predict(cv2img)
            retLabel = self.labels_dict_rev.get(label)
            if not retLabel:
                retLabel = label
            return (retLabel, confidence)

        retVal = []
        for image in images:
            cv2img = cv2.cvtColor(image, cv2.cv.CV_BGR2GRAY)
            label, confidence = self.model.predict(cv2img)
            retLabel = self.labels_dict_rev.get(label)
            if not retLabel:
                retLabel = label
            retVal.append((retLabel, confidence))
        return retVal

    # def update():
    # OpenCV 2.4.4 doens't support update yet. It asks to train.
    #     But it's not updating it.
    #     Once OpenCV starts supporting update, this function should be added
    #     it can be found at https://gist.github.com/jayrambhia/5400347

    def save(self, filename):
        """
        **SUMMARY**
        Save the trainging data.
        **PARAMETERS**
        * *filename* - File where you want to save the data.
        **RETURNS**
        Nothing. None.
        **EXAMPLES**
        >>> f = FaceRecognizer()
        >>> imgs1 = ImageSet(path/to/images_of_type1)
        >>> labels1 = LabelSet("type1", imgs1)
        >>> imgs2 = ImageSet(path/to/images_of_type2)
        >>> labels2 = LabelSet("type2", imgs2)
        >>> imgs3 = ImageSet(path/to/images_of_type3)
        >>> labels3 = LabelSet("type3", imgs3)
        >>> imgs = concatenate(imgs1, imgs2, imgs3)
        >>> labels = concatenate(labels1, labels2, labels3)
        >>> f.train(imgs, labels)
        >>> imgs = ImageSet("path/to/testing_images")
        >>> print f.predict(imgs)
        #Save New Fisher Training Data
        >>> f.save("new_trainingdata.xml")
        """
        if not self.supported:
            warnings.warn("Fisher Recognizer is supported by OpenCV >= 2.4.4")
            return None

        self.model.save(filename)

    def load(self, filename):
        """
        **SUMMARY**
        Load the trainging data.
        **PARAMETERS**
        * *filename* - File where you want to load the data from.
        **RETURNS**
        Nothing. None.
        **EXAMPLES**
        >>> f = FaceRecognizer()
        >>> f.load("trainingdata.xml")
        >>> imgs = ImageSet("path/to/testing_images")
        >>> print f.predict(imgs)
        """
        if not self.supported:
            warnings.warn("Fisher Recognizer is supported by OpenCV >= 2.4.4")
            return None

        self.model.load(filename)
        loadfile = open(filename, "r")
        for line in loadfile.readlines():
            if "cols" in line:
                match = re.search("(?<=\>)\w+", line)
                tsize = int(match.group(0))
                break
        loadfile.close()
        w = int(tsize ** 0.5)
        h = tsize / w
        while (w * h != tsize):
            w += 1
            h = tsize / w
        self.imageSize = (w, h)


class ImageSet(list):
    """
    **SUMMARY**
    This is a class derived from list to keep a list of images. It helps
    in loading all the images from a given directory. Bulk operations can
    be performed on this class. eg. cropping faces
    **EXAMPLES**
    >>> imgs = ImageSet("directory_with_images")
    >>> imgs.show()
    """

    def __init__(self, directory="", imgs=None):
        if os.path.isfile(directory):
            img = cv2.imread(directory)
            self.append(img)
            return
        if imgs:
            self.extend(imgs)
            return
        try:
            imagefiles = os.listdir(directory)

        except OSError as error:
            print "OS Error({0}): {1}".format(error.errno, error.strerror)
            warnings.warn("encountered the above mentioned error. Returning Empty list.")
            return
        for imagefile in imagefiles:
            filename = os.path.join(directory, imagefile)
            img = cv2.imread(filename)
            if isinstance(img, np.ndarray):
                self.append(img)

    def cropFaces(self, cascade=None):
        """
        **SUMMARY**
        This function helps in cropping a certain object in all of the images
        by using the provided haar cascade. A haar classifier is implemented
        and images are cropped. This function also supports multiple faces in
        one single image.
        **PARAMETERS**
        cascade - haar cascade string
        **RETURNS**
        ImageSet
        **EXAMPLES**
        >>> imgs = ImageSet("some_directory/")
        >>> cascade = "face.xml"
        >>> faces = imgs.cropFaces(cascade)
        >>> faces.show()
        """
        if not cascade:
            cascade = CASCADE_PATH
        else:
            if not os.path.isfile(cascade):
                warnings.warn("The provided cascade does not exist. Using default cascade")
                cascade = CASCADE_PATH
        classifier = cv2.CascadeClassifier(cascade)
        gray = [cv2.cvtColor(img, cv2.cv.CV_BGR2GRAY) for img in self]
        objects = [classifier.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3, minSize=(10, 10),
                                               flags=cv2.cv.CV_HAAR_SCALE_IMAGE) for img in gray]
        imgs = [[img[ob[1]:ob[1] + ob[3], ob[0]:ob[0] + ob[2]] for ob in obj] for img, obj in zip(self, objects) if
                isinstance(obj, np.ndarray)]
        imgs = concatenate(*imgs)
        return ImageSet(imgs=imgs)

    def show(self, name="facerec", delay=500):
        """
        **SUMMARY**
        This function uses OpenCV's default display utilities and shows
        all the images present in the ImageSet.
        **PARAMETERS**
        name  - Name of the display window
        delay - Time delay between each images in milliseconds.
        **EXAMPLES**
        >>> imgs = ImageSet("images/")
        >>> imgs.show("faces", 2000)
        """
        for img in self:
            cv2.imshow(name, img)
            cv2.waitKey(delay)
        cv2.destroyWindow(name)


class LabelSet(list):
    """
    **SUMMARY**
    This is a class derived from list to implement rapid label
    generation for a given ImageSet. labels can be integers or
    string variables.
    ***PARAMETERS***
    label    - Label name
    imageset - ImageSet which is to be labelled.
    **EXAMPLES**
    >>> imgs = ImageSet("images/")
    >>> labels = LabelSet("positive", imgs)
    """

    def __init__(self, label, imageset):
        if not isinstance(imageset, collections.Iterable):
            warnings.warn("The provided ImageSet is not a list")
            return None
        labels = [label] * len(imageset)
        self.extend(labels)


def concatenate(*args):
    """
    **SUMMARY**
    This function is implemented for rapid concatenation of
    images and labels.
    **EXAMPLES**
    >>> f = FaceRecognizer()
    >>> imgs1 = ImageSet(path/to/images_of_type1)
    >>> labels1 = LabelSet("type1", imgs1)
    >>> imgs2 = ImageSet(path/to/images_of_type2)
    >>> labels2 = LabelSet("type2", imgs2)
    >>> imgs3 = ImageSet(path/to/images_of_type3)
    >>> labels3 = LabelSet("type3", imgs3)
    >>> imgs = concatenate(imgs1, imgs2, imgs3)
    >>> labels = concatenate(labels1, labels2, labels3)
    >>> f.train(imgs, labels)
    >>> imgs = ImageSet("path/to/testing_images")
    >>> print f.predict(imgs)
    """
    retVal = []
    for arg in args:
        retVal.extend(arg)
    return retVal


def plot_training_results():
    ''' male tests: 30 male 20 female
     male tests: 17 male 33 female

     200 male 200 female
      male tests: 33 male 17 female
     male tests: 14 male 36 female

    400 ,400
     male tests: 37 male 13 female
     female tests: 11 male 39 female
    '''
    results = [{'n': 100, 'male': [30, 20], 'female': [17, 33]},
               {'n': 200, 'male': [33, 17], 'female': [14, 36]},
               {'n': 200, 'male': [37, 13], 'female': [11, 39]}]


if __name__ == "__main__":
    # %Use CSV files for training
    f = FaceRecognizer()
    csvfile = 'genders_small.csv'
    f.train(csvfile=csvfile, delimiter=";")
    imgs = ImageSet("test/male")
    predictions = f.predict(imgs)
    print predictions
    sumzero = 0
    sumone = 0
    for p in predictions:
        a = p[0]
        if a == '0':
            sumzero += 1
        elif a == '1':
            sumone += 1
        else:
            print('apparently there is a third category')
    print(' male tests: {0} male {1} female').format(sumzero, sumone)
    imgs = ImageSet("test/female")
    predictions = f.predict(imgs)
    print predictions
    sumzero = 0
    sumone = 0
    for p in predictions:
        a = p[0]
        if a == '0':
            sumzero += 1
        elif a == '1':
            sumone += 1
        else:
            print('apparently there is a third category')
    print(' female tests: {0} male {1} female').format(sumzero, sumone)

'''
100 male 100 female training
 male tests: 30 male 20 female
[('1', 116.23974819420908), ('0', 340.7092253217653), ('1', 70.50905492730237), ('1', 48.380605709878864), ('0', 348.91221131260374), ('1', 412.5888333569815), ('0', 302.9549901597826), ('0', 374.97081887102337), ('1', 24.659717990867705), ('0', 202.49719301537013), ('1', 182.87517147942106), ('1', 218.58077798395837), ('0', 48.647145784005374), ('0', 105.62521739403752), ('1', 159.12621368444036), ('1', 17.402342675975945), ('1', 81.10703837742744), ('1', 0.03724994130197956), ('1', 117.97090503714833), ('1', 390.1728318084548), ('0', 3.711439232941018), ('1', 147.19843637413055), ('1', 0.011998134078339717), ('1', 87.64322790033441), ('1', 349.9851030255048), ('0', 26.227887299907366), ('1', 19.635440697707395), ('1', 27.5011432121226), ('1', 228.51771056136243), ('1', 24.689039843660737), ('0', 340.0522336003684), ('1', 95.55182282654482), ('0', 227.68134530275935), ('1', 393.62653527769817), ('1', 19.68963047978366), ('1', 14.9901193420161), ('1', 4.211470690451392), ('0', 463.84915540961674), ('1', 338.56960954314366), ('0', 158.30585048244274), ('1', 128.6733958492527), ('1', 181.42017850876596), ('1', 309.7681826065375), ('1', 279.7553137958643), ('0', 1.3456278819319891), ('0', 372.28603093259466), ('0', 30.00850442782263), ('1', 48.125850423988254), ('0', 159.14321776510516), ('1', 356.4525228616915)]
 male tests: 17 male 33 female

 200 male 200 female
  male tests: 33 male 17 female
[('1', 2.5934019522761673), ('1', 137.29288142277161), ('0', 60.83035691917786), ('1', 32.55417242231181), ('1', 217.30227548524894), ('1', 0.034765068784906816), ('1', 55.64650981831329), ('1', 266.3096285205563), ('0', 133.0442522141666), ('0', 14.037216813618045), ('1', 132.30541962131736), ('1', 57.16857072124219), ('1', 221.54997662791095), ('0', 265.7451827950006), ('1', 95.77079616598647), ('1', 225.53742305283822), ('1', 35.93368303940031), ('1', 23.538237570537206), ('1', 12.64785434378004), ('1', 125.73344889441842), ('0', 102.33768517969645), ('1', 184.66882196686353), ('1', 206.72569388542504), ('1', 107.13404430371457), ('1', 170.30811649272096), ('1', 107.13340102593818), ('1', 351.73111728892536), ('1', 31.253879547249085), ('1', 125.43259438515082), ('1', 7.498631895866197), ('1', 199.35223434026193), ('1', 204.5560274725533), ('0', 215.9770318937397), ('1', 194.30043473848374), ('1', 6.709953699163577), ('1', 128.8601392318581), ('1', 330.4628514943971), ('0', 18.97888808254345), ('0', 158.1506474660268), ('1', 37.14340029277977), ('1', 16.234818687922427), ('0', 202.71681671833662), ('0', 139.0476004114962), ('1', 106.02115296318587), ('0', 0.21013450780162657), ('1', 231.1603613057846), ('0', 143.68119173334532), ('1', 295.27432370385554), ('0', 7.762226317948091), ('0', 201.56831548363544)]
 male tests: 14 male 36 female

400 ,400
 male tests: 37 male 13 female
 female tests: 11 male 39 female

600,600\
 male tests: 34 male 16 female
 male tests: 11 male 39 female


800,800
e tests: 29 male 21 female
 male tests: 10 male 40 female



'''''