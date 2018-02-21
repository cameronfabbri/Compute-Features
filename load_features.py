'''

   Base file for using the features that were computed and stored in the pickle file.
   This is tensorflow agnostic

   Ex. python load_features.py inception_v1_features.pkl

'''

import numpy as np
import sys
import cPickle as pickle

if __name__ == '__main__':

   try:
      pkl_file = open(sys.argv[1], 'rb')
      features = pickle.load(pkl_file)
   except:
      print 'Must provide a pickle file'
      exit()

   for image, feature in features.iteritems():
      print image, ':', feature
      exit()

      # do whatever you want with features

