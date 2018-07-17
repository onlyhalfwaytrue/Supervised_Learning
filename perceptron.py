#perceptron.py
#-------------

#Perceptronimplementation
import util
from time import time
PRINT=True

class PerceptronClassifier:
	"""
	Perceptronclassifier.
	"""
	def __init__(self,legalLabels,max_iterations):
		self.legalLabels=legalLabels
		self.type="perceptron"
		self.max_iterations=max_iterations
		self.weights=util.Counter();
		for label in legalLabels:
			self.weights[label]=util.Counter()

	def setWeights(self,weights):
		assert len(weights)==len(self.legalLabels);
		self.weights=weights;


	def train(self,trainingData,trainingLabels,validationData,validationLabels):
		for iteration in range(self.max_iterations):
			print "Starting iteration",iteration,"...";
			init = time();
			for i in range(len(trainingData)):
				y = trainingLabels[i];
				y_prime = self.classify([trainingData[i]])[0];
				newWeights = self.weights;
				if y_prime==y:
					pass;
				else:
					newWeights[y] = self.weights[y] + trainingData[i];
					newWeights[y_prime] = self.weights[y_prime] - trainingData[i];
					self.setWeights(newWeights);
			final = time();
			print "Seconds elapsed for this iteration:", final-init;




	def classify(self,data):
		guesses=[]
		for datum in data:
			vectors = util.Counter()
			for l in self.legalLabels:
				vectors[l]=self.weights[l]*datum;
			guesses.append(vectors.argMax());
		return guesses


	def findHighWeightFeatures(self,label):
		featuresWeights=[];
		c = util.counter();

		for i in range(28):
			for j in range(28):
				c[(i,j)]=self.weights[label][(i,j)];

		for k in range(100):
			x=c.argMax();
			featuresWeights.append(x);
			c[x]=-(self.max_iterations+1);
			return featuresWeights
