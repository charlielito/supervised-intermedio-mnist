'''
SVM and KNearest digit recognition.
Following preprocessing is applied to the dataset:
 - Moment-based image deskew (see deskew())
 - Digit images are split into 4 10x10 cells and 16-bin
   histogram of oriented gradients is computed for each
   cell
 - Transform histograms to space with Hellinger metric (see [1] (RootSIFT))
[1] R. Arandjelovic, A. Zisserman
    "Three things everyone should know to improve object retrieval"
    http://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf
'''

import numpy as np
import cv2
from multiprocessing.pool import ThreadPool
from numpy.linalg import norm
from dataget import data # <== datage

SZ = 28 # size of each digit is SZ x SZ
CLASS_N = 10


def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

class StatModel(object):
    def load(self, fn):
        self.model.load(fn)
    def save(self, fn):
        self.model.save(fn)

class KNearest(StatModel):
    def __init__(self, k = 3):
        self.k = k
        self.model = cv2.ml.KNearest_create()

    def train(self, samples, responses):
        self.model = cv2.ml.KNearest_create()
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        retval, results, neigh_resp, dists = self.model.findNearest (samples, self.k)
        return results.ravel()

class SVM(StatModel):
    def __init__(self, C = 1, gamma = 0.5):
        self.params = dict( kernel_type = cv2.ml.SVM_RBF,
                            svm_type = cv2.ml.SVM_C_SVC,
                            C = C,
                            gamma = gamma )
        self.model = cv2.ml.SVM_create()
        self.model.setType(cv2.ml.SVM_C_SVC)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setGamma(gamma)
        self.model.setC(C)

    def train(self, samples, responses):
        self.model = cv2.ml.SVM_create()
        self.model.setType(cv2.ml.SVM_C_SVC)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setGamma(self.params['gamma'])
        self.model.setC(self.params['C'])
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses )

    def predict(self, samples):
        x = self.model.predict(samples)[1].ravel()
        return x




def evaluate_model(model, samples, labels):
    resp = model.predict(samples)
    err = (labels != resp).mean()
    print 'error: %.2f %%' % (err*100)

    confusion = np.zeros((10, 10), np.int32)
    for i, j in zip(labels, resp):
        confusion[int(i), int(j)] += 1
    print 'confusion matrix:'
    print confusion
    print


def preprocess_simple(digits):
    return np.float32(digits).reshape(-1, SZ*SZ) / 255.0

def preprocess_hog(digits):
    samples = []
    for img in digits:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        bin_n = 16
        bin = np.int32(bin_n*ang/(2*np.pi))
        bin_cells = bin[:10,:10], bin[10:,:10], bin[:10,10:], bin[10:,10:]
        mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)

        # transform to Hellinger kernel
        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= norm(hist) + eps

        samples.append(hist)
    return np.float32(samples)


if __name__ == '__main__':
    print __doc__

    dataset = data("mnist").get()

    digits, labels_train = dataset.training_set.arrays()
    digits_test, labels_test = dataset.test_set.arrays()

    print 'preprocessing...'

    digits_deskew = map(deskew, digits)
    samples = preprocess_hog(digits_deskew)


    samples_train = np.float32(map(np.ravel,digits)) #error 2,98% k =3
    samples_test = np.float32(map(np.ravel,digits_test))


    samples_train = np.float32(map(np.ravel,digits_deskew)) #error 2% k =3
    digits_deskewtest = map(deskew, digits_test)
    samples_test = np.float32(map(np.ravel,digits_deskewtest))


    '''
    print 'training KNearest...'
    model = KNearest(k=3)
    print samples_train.shape, labels_train.shape
    model.train(samples_train, labels_train)
    evaluate_model(model, samples_test, labels_test)

    '''
    #samples_train = samples #error 4.93% k =3
    #samples_test = preprocess_hog(digits_deskewtest)


    print 'training SVM...'
    model = SVM(C=5, gamma=0.05)
    model.train(samples_train, labels_train)
    vis = evaluate_model(model, samples_test, labels_test)
    print 'saving SVM as "digits_svm.dat"...'
    model.save('digits_svm.dat')
