import tensorflow as tf 
import model
import read
import numpy as np
import cv2
import sys


print("loading data")
labels, query, passage = read.return_vectors()

clip = int(0.7*len(labels))
tot = len(labels)

train_labels = labels[:clip]
train_query = query[:clip]
train_passage = passage[:clip]

val_labels = labels[clip : tot]
val_query = query[clip : tot]
val_passage = passage[clip : tot]

print("Data loaded")

net = model.Model()
print("Build model")

net.build_model()
print("Build Successful")

print("Training with training examples : " + str(clip) + " , validation : " + str(tot - clip))
net.train_model(train_query, train_passage, train_labels, val_query, val_passage, val_labels, 30)
