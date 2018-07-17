from node import *
import util
import math
from random import *


class HiddenLayer:
	def __init__(self,num):
		self.num = num;
		'''if self.num == 98:
			self.nodes = self.makeHiddenNodes98(num);
		else:'''
		self.nodes = self.makeHiddenNodes(num);
		self.outputGrid = util.Counter();

	def makeHiddenNodes(self,number):
		'''shift = int(math.sqrt(784/number));
		squares = int(28/shift);'''
		nodeCounter = util.Counter();
		squares = int(math.sqrt(number));
		for i in range(number):
			nodeCounter[i] = self.hiddenNode();
		return nodeCounter;

	def hiddenNode(self):
		newNode = Node(1,0);
		newWeights = util.Counter();
		for i in range(28):
			for j in range (28):
				#if the performance is bad replace with newWeights[(i,j)] = uniform(-0.5,0.5)
				if (j+i) % 2 == 1:
					newWeights[(i,j)] = uniform(-0.5,0);
				else:
					newWeights[(i,j)] = uniform(0,0.5);
		newNode.updateWeights(newWeights);
		return newNode;
	'''
	def makeHiddenNodes98(self,number):
		shift = int(math.sqrt(2*784/number));
		rects = int(28/shift);
		nodeCounter = util.Counter();
		for i in range(2*rects):
			for j in range(rects):
				nodeCounter[(i,j)] = self.hiddenNode98((i,j),shift);
		return nodeCounter;

	def hiddenNode98(self,coord,shift):
		newNode = Node(1,-1);
		newWeights = util.Counter();
		focusfeatures = [];

		for f in range(shift/2):
			for g in range(shift):
				focusfeatures.append((coord[0]+f,coord[1]+g));

		for i in range(28):
			for j in range (28):
				if (i,j) in focusfeatures:
					newWeights[(i,j)] = uniform(0.85,0.99);
				else:
					newWeights[(i,j)] = uniform(0,0.01);
		newNode.updateWeights(newWeights);
		return newNode;'''

	def updateBiases(self, new_biases_vector):
		for key in new_biases_vector:
			self.nodes[key].bias = new_biases_vector[key];

	def updateLayerWeights(self,new_weights_vector):
		for key in new_weights_vector.keys():
			self.nodes[key].updateWeights(new_weights_vector[key]);

	def activateLayer(self,inputs):
		grid = int(math.sqrt(self.num));
		for key in self.nodes.keys():
			self.nodes[key].activate(inputs);
			self.outputGrid[key] = self.nodes[key].output;
