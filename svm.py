# svm.py
# -------------

# svm implementation
import util
#import numpy as np
#from sklearn.svm import LinearSVC

PRINT = True

class SVMClassifier:
  """
  svm classifier 
  In train you need to make a 2d array, where the first dimension is the number
  of training data and the second dimension is the pixel values of that training Data
Outside of the for loop within train, you use self. (Classifier name).fit (that array 2d array you
 just made , the training label array) Finally in classify you just use guesses=self.(Classifier  name). 
 predict(2d array like the one you needed on training)
  """
  def __init__( self, legalLabels):
    self.legalLabels = legalLabels
    self.type = "svm"
    self.name = LinearSVC(1)	
  def train( self, trainingData, trainingLabels, validationData, validationLabels ):
    A = np.array([[x for x in trainingData.size] for y in trainingData.winfo_pixels()])
    #puzzle = np.array([[Node(0,j,i) for j in range(dim)] for i in range(dim)])
    print "Starting iteration ", iteration, "..."
    for i in range(len(trainingData)):
      pass  
    
  def classify(self, data ):
    guesses = []
    for datum in data:
      # fill predictions in the guesses list
      "*** YOUR CODE HERE ***"
      util.raiseNotDefined()
      
    return guesses

