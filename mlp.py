#mlp.py
#-------------

#mlpimplementation
import util
import math
from time import time
from random import *
from node import *
from hiddenlayer import *
from outputlayer import *
PRINT=True

class MLPClassifier:
	def __init__(self,legalLabels,max_iterations):
		self.legalLabels=legalLabels;
		self.type="mlp";
		self.max_iterations=max_iterations;
		self.hLayer = HiddenLayer(78);
		self.oLayer = OutLayer(self.hLayer,self.legalLabels);

	def deltas_at_output_layer(self,expected_values_vector):
		delta_vector = util.Counter();
		for i in self.oLayer.legalLabels:
			g_o_prime = (self.oLayer.nodes[i].act(inLayer = None,take_derivative = True)[0]);
			#print "value of output layer node ", i, " activation: ", self.oLayer.nodes[i].activation;
			delta_vector[i] = (self.oLayer.nodes[i].output-expected_values_vector[i])*g_o_prime;
			#print "derivative, delta of output node ",i," activation function: ", g_o_prime, ", ", delta_vector[i] ;
		return delta_vector #vector is a counter with 10 entries which are errors at each node;

	def deltas_at_hidden_layer(self,dels_at_output_layer):
		delta_vector = util.Counter();
		for key in self.hLayer.nodes.keys():
			summer = 0.0;
			for i in self.oLayer.legalLabels:
				summer += self.oLayer.nodes[i].weights[key]*dels_at_output_layer[i];
			delta_vector[key] = (self.hLayer.nodes[key].act(inLayer = None,take_derivative = True)[0])*summer;
		return delta_vector;

	def new_weights_for_output_layer(self,dels_at_output_layer,eta):
		new_weights = util.Counter();
		new_biases = util.Counter();
		for i in self.oLayer.legalLabels:
			i_weight = util.Counter();
			current = self.oLayer.nodes[i];
			for key in current.weights.keys():
				old = current.weights[key];
				new = old - (eta*dels_at_output_layer[i]*self.hLayer.nodes[key].output);
				i_weight[key] = new;
			old_bias = current.bias;
			new_bias = old_bias - (eta*0.9)*dels_at_output_layer[i];
			new_biases[i] = new_bias;
			#print "new bias for out node ", i,": ", new_bias;
			new_weights[i] = i_weight;
		self.oLayer.updateLayerWeights(new_weights);
		self.oLayer.updateBiases(new_biases)

	def new_weights_for_hidden_layer(self,dels_at_hidden_layer,inputLayer,eta):
		new_weights = util.Counter();
		new_biases = util.Counter();
		for key in self.hLayer.nodes.keys():
			key_weights = util.Counter();
			current = self.hLayer.nodes[key];
			for key2 in self.hLayer.nodes[key].weights.keys():
				old = current.weights[key2];
				new = old - (eta*dels_at_hidden_layer[key]*inputLayer[key2]);
				key_weights[key2] = new;
			old_bias = current.bias;
			new_bias = old_bias - (eta*0.9)*dels_at_hidden_layer[key];
			new_biases[key] = new_bias;
			new_weights[key] = key_weights;
		self.hLayer.updateLayerWeights(new_weights);
		self.hLayer.updateBiases(new_biases);

	def train(self,trainingData,trainingLabels,validationData,validationLabels):
		def learning_rate(init, final, num_decreases, num_examples):
			current = init;
			learning_rate_vector = [];
			decrease_interval = int(num_examples/num_decreases);
			decrease_size = (init-final)/num_decreases;
			for it in range(num_examples):
				if (it+1)%decrease_interval == 0:
					current = current-decrease_size;
				learning_rate_vector.append(current);
			return learning_rate_vector;

		for iteration in range(self.max_iterations):
			print "Starting iteration",iteration,"..."
			learning = learning_rate(0.1,0.08,50,int(len(trainingLabels)));
			init = time();
			for i in range(len(trainingData)):
				#print "Training Item ",i," ";
				y=trainingLabels[i];
				y_prime=self.classify([trainingData[i]])[0];
				if y == y_prime:
					#print "CORRECT!"
					pass;
				else:
					#print "ERROR, t vs. f:",y, " ", y_prime;
					L_over_2 = 1;
					y_vector=[-1*L_over_2 for i in range(10)];
					y_vector[y]=L_over_2;
					out_deltas = self.deltas_at_output_layer(y_vector);
					hidden_deltas = self.deltas_at_hidden_layer(out_deltas);
					self.new_weights_for_output_layer(out_deltas,learning[i]);
					self.new_weights_for_hidden_layer(hidden_deltas,trainingData[i],learning[i]);
			final = time();
			print "Seconds elapsed for this iteration:", final-init;

	def classify(self,data):
		guesses=[];
		for datum in data:
			self.hLayer.activateLayer(datum);
			self.oLayer.activateLayer();
			guesses.append(self.oLayer.outputValue());
		return guesses
