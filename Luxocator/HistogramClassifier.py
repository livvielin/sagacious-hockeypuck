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
