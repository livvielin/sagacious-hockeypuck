import numpy
import cv2
import scipy.io
import scipy.sparse

class HistogramClassifier(object):

  def __init__(self):

    # controls level of logging
    self.verbose = False
    # public float that defines similarity threshold
    # if all average similarities fall below, query image given unknown classification
    self.minimumSimilarityForPositiveLabel = 0.075

    # variables related to color space
    self._channels = range(3)
    self._histSize = [256] * 3
    self._ranges = [0, 255] * 3
    # dictionary to map string keys to lists of reference histograms
    self._references = {}

    # function to create normalized histogram
    # optionally converts histogram to sparse format
    def _createNormalizedHist(self, image, sparse):
        # Create the histogram
        hist = cv2.calcHist([image], self._channels, None, self._histSize, self._ranges)
        # Normalize the histogram
        hist[:] = hist * (1.0 / numpy.sum(hist))
        # Convert the histogram to one column for efficient storage
        hist = hist.reshape(16777216, 1)
        if sparse:
            #Conver the histogram to a sparse matrix
            hist = scipy.sparse.csc_matrix(hist)
        return hist
