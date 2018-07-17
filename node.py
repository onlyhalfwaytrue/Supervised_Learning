import util
import math
from random import *

class Node:
	def __init__(self,layer,bias=-1):
		self.layer=layer;
		self.weights=util.Counter();
		self.activation=0;
		self.output=0;
		self.bias = bias;

	def updateWeights(self,newWeights):
		self.weights = newWeights;

	def act(self,inLayer = None,take_derivative = False):
		ret_vector = [];
		if self.layer == 1:
			L = 2;
			k = 0.07;
			b = self.bias;
			"***ACTIVATION FUNCTION FOR HIDDEN LAYER NODES***"
			if take_derivative == True:
				val = self.activation;
				numerator = k*L*math.exp(k*(val));
				denominator = (1+math.exp(k*(val)))**2;
				ret_vector.append(numerator/denominator);
				#print "derivative taken: ", ret_vector[0], " "
				return ret_vector;
			else:
				ret_vector.append(self.weights * inLayer + self.bias);
				denom = 1 + math.exp(-k*(ret_vector[0]));
				ret_vector.append(L/denom - L/2);
				return ret_vector;
		else:
			"***ACTIVATION FUNCTION FOR OUTPUT LAYER NODES***"
			L = 2;
			k = 2;
			b = self.bias;
			if take_derivative == True:
				val = self.activation;
				numerator = k*L*math.exp(k*(val));
				denominator = (1+math.exp(k*(val)))**2;
				ret_vector.append(numerator/denominator);
				#print "derivative taken: ", ret_vector[0], " "
				return ret_vector;
			else:
				ret_vector.append(self.weights * inLayer.outputGrid + self.bias);
				denom = 1 + math.exp(-k*(ret_vector[0]));
				ret_vector.append(L/denom - L/2);
				return ret_vector;


	def activate(self,inLayer):
		if self.layer == 1:
			"***ACTIVATION FUNCTION FOR HIDDEN LAYER NODES***"
			response = self.act(inLayer);
			self.activation = response[0];
			self.output = response[1];
		else:
			"***ACTIVATION FUNCTION FOR OUTPUT LAYER NODES***"
			response = self.act(inLayer);
			self.activation = response[0];
			self.output = response[1];
