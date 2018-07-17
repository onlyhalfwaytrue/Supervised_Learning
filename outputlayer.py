from node import *
import util
import math
from random import *
from hiddenlayer import *


class OutLayer:
	def __init__(self,inLayer,legalLabels):
		self.inLayer = inLayer;
		self.legalLabels = legalLabels;
		self.nodes = self.makeOutputNodes();
		self.outputList = util.Counter();

	def makeOutputNodes(self):
		outNodes = util.Counter();
		for i in self.legalLabels:
			outNodes[i] = self.outNode(i);
		return outNodes;

	def outNode(self,num):
		newNode = Node(2,0);
		newWeights = util.Counter();
		m = 0;
		for key in self.inLayer.nodes.keys():
			newWeights[key] = uniform(0,0.01);
			#if m % 2 == 0:
			#	newWeights[key] = uniform(-0.5,0);
			#else:
			#	newWeights[key] = uniform(0,0.5);
			#m+=1;
		newNode.updateWeights(newWeights);
		return newNode;

	def activateLayer(self):
		for i in self.legalLabels:
			self.nodes[i].activate(self.inLayer);
			self.outputList[i] = self.nodes[i].output;

	def updateBiases(self, new_biases_vector):
		for key in new_biases_vector:
			self.nodes[key].bias = new_biases_vector[key];

	def outputValue(self):
		return self.outputList.argMax();


	def updateLayerWeights(self, new_weights_vector):
		for i in new_weights_vector.keys():
			self.nodes[i].updateWeights(new_weights_vector[i]);
