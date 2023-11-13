from sklearn.metrics import DistanceMetric
from sklearn.svm import SVC
from skimage.feature import local_binary_pattern
import numpy as np

class LPBH_SVM_Recognize():
    def __init__(self, C=100, Gamma=0.001):
        self.svm = SVC(kernel='precomputed', C=C, gamma=Gamma)
        self.chi2 = DistanceMetric.get_metric('pyfunc', func=self.chi2_distance)
        self.face_histograms = []
        self.hist_mat = []
    def chi2_distance(self, hist1, hist2, gamma=0.5):
        chi = - gamma * np.sum(((hist1 - hist2) ** 2) / (hist1 + hist2 + 1e-7))
        return chi

    def find_lbp_histogram(self, image, P=8, R=1, eps=1e-7, n_window=(8, 8)):
        E = []
        h, w = 100, 100
        h_sz = int(np.floor(h / n_window[0]))
        w_sz = int(np.floor(w / n_window[1]))
        lbp_img = local_binary_pattern(image, P=P, R=R, method="default")
        for (x, y, C) in self.sliding_window(lbp_img, stride=(h_sz, w_sz), window=(h_sz, w_sz)):
            if C.shape[0] != h_sz or C.shape[1] != w_sz:
                continue
            H = np.histogram(C,
                             bins=2 ** P,
                             range=(0, 2 ** P),
                             density=True)[0]

            H = H.astype("float")
            H /= (H.sum() + eps)
            E.extend(H)
        return E

    def sliding_window(self, image, stride, window):
        for y in range(0, image.shape[0], stride[0]):
            for x in range(0, image.shape[1], stride[1]):
                yield (x, y, image[y:y + window[1], x:x + window[0]])

    def train(self, x, y):
        self.face_histograms = [self.find_lbp_histogram(img) for img in x]
        self.hist_mat = np.array(self.face_histograms, dtype=np.float32)
        K = self.chi2.pairwise(self.hist_mat, self.hist_mat)
        self.svm.fit(K, y)

    def predict(self, x):
        hists = [self.find_lbp_histogram(img) for img in x]
        hist_mat = np.array(hists, dtype=np.float32)
        K = self.chi2.pairwise(hist_mat, self.hist_mat)
        idx = self.svm.predict(K)
        return idx, None