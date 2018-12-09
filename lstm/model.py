from __future__ import unicode_literals
import tensorflow as tf
import numpy as np
import cv2
import os

embed_size = 50

class Model(object):
	def __init__(self):
		self.train_phase = True
		self.batch_size = 10
		self.query_steps = 12
		self.passage_steps = 50
		self.features = embed_size
		self.l_rate = 1e-4
		self.num_classes = 10

	def model(self, in_query, in_passage):
		self.basic_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units = 50)

		outputs, states = tf.nn.dynamic_rnn(self.basic_cell, in_query, dtype = tf.float32)
		outputs = tf.unstack(tf.transpose(outputs, perm = [1, 0, 2]))
		last_output = outputs[-1]
		query_vec = tf.contrib.layers.fully_connected(last_output, 100, activation_fn = None)
		# print "query_vec", query_vec

		outputs, states = tf.nn.dynamic_rnn(self.basic_cell, in_passage[:, 0], dtype = tf.float32)
		outputs = tf.unstack(tf.transpose(outputs, perm = [1, 0, 2]))
		last_output = outputs[-1]
		passage_vec = tf.contrib.layers.fully_connected(last_output, 100, activation_fn = None)
		vector = tf.concat([query_vec, passage_vec], 1)
		logits = tf.contrib.layers.fully_connected(vector, 1, activation_fn = None)


		for i in range(1, 10):
			outputs, states = tf.nn.dynamic_rnn(self.basic_cell, in_passage[:, i], dtype = tf.float32)
			outputs = tf.unstack(tf.transpose(outputs, perm = [1, 0, 2]))
			last_output = outputs[-1]
			passage_vec = tf.contrib.layers.fully_connected(last_output, 100, activation_fn = None)
			vector = tf.concat([query_vec, passage_vec], 1)
			this_out = tf.contrib.layers.fully_connected(vector, 1, activation_fn = None)
			logits = tf.concat([logits, this_out], 1)
		print logits
		return logits

	def build_model(self):

		self.in_query = tf.placeholder(tf.float32, [None, self.query_steps, self.features])
		self.in_passage = tf.placeholder(tf.float32, [None, 10, self.passage_steps, self.features])
		self.labels = tf.placeholder(tf.float32, [None, self.num_classes])

		self.logits = self.model(self.in_query, self.in_passage)

		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.logits, labels = self.labels))

		self.optimizer = tf.train.AdamOptimizer(learning_rate = self.l_rate, beta1 = 0.5).minimize(self.loss)

		correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.labels, 1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


	def train_model(self, train_query, train_passage, train_labels, val_query, val_passage, val_labels, epochs):
		init = tf.global_variables_initializer()
		max_acc = 0
		file = open("log.txt",'w')
		# print len(train_query)
		# exit()

		with tf.Session() as sess:
			sess.run(init)
			with tf.device('/gpu:0'):
				for epoch in range(epochs):
					for i in range(0, len(train_query), self.batch_size):
						if(i + self.batch_size < len(train_query)):
							l = train_labels[i : i + self.batch_size]
							q = train_query[i : i + self.batch_size]
							p = train_passage[i : i + self.batch_size]
						else:
							l = train_labels[i : len(train_query)]
							q = train_query[i : len(train_query)]
							p = train_passage[i : len(train_query)]

						feed_dict = {self.in_query : q, self.in_passage : p, self.labels : l}
						sess.run(self.optimizer, feed_dict)

					print "Loss = ", sess.run(self.loss, feed_dict)
					feed_dict = {self.in_query : val_query, self.in_passage : val_passage, self.labels : val_labels}
					# print "Validation ", sess.run(self.logits, feed_dict)
					acc = sess.run(self.accuracy, feed_dict)

					print("Epoch " + str(epoch + 1) + " Accuracy : " + str(acc))
					file.write("Epoch : " + str(epoch + 1) + ", Accuracy : " + str(acc) + "\n")

					if(acc > max_acc):
						max_acc = acc
						with tf.device('/cpu:0'):
							saver = tf.train.Saver()
							if(not(os.path.exists("weights/" + str(epoch) + "/"))):
								os.makedirs("weights/"+str(epoch)+"/")
							save_path = saver.save(sess,"weights/"+str(epoch)+"/weight.cpkt")
							print("saved")


if __name__ == "__main__":
	net = Model()
	net.build_model()