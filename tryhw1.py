from uwnet import *

def conv_net():
  #Original
  l = [   make_convolutional_layer(32, 32, 3, 8, 3, 1),
           make_activation_layer(RELU),
           make_maxpool_layer(32, 32, 8, 3, 2),
           make_convolutional_layer(16, 16, 8, 16, 3, 1),
           make_activation_layer(RELU),
           make_maxpool_layer(16, 16, 16, 3, 2),
           make_convolutional_layer(8, 8, 16, 32, 3, 1),
           make_activation_layer(RELU),
           make_maxpool_layer(8, 8, 32, 3, 2),
           make_convolutional_layer(4, 4, 32, 64, 3, 1),
           make_activation_layer(RELU),
           make_maxpool_layer(4, 4, 64, 3, 2),
           make_connected_layer(256, 10),
           make_activation_layer(SOFTMAX)]


  #To TEST FOR BETTER Convolutional Architecture (Need to change values and add/remove layers):
  #l = [   make_convolutional_layer(32, 32, 3, 8, 3, 2),
  #  make_activation_layer(RELU),
  #  make_maxpool_layer(32, 32, 8, 3, 2),
  #  make_convolutional_layer(16, 16, 8, 16, 3, 2),
  #  make_activation_layer(RELU),
  #  make_maxpool_layer(16, 16, 16, 3, 2),
  #  make_convolutional_layer(8, 8, 16, 32, 3, 2),
  #  make_activation_layer(RELU),
  #  make_maxpool_layer(8, 8, 32, 3, 2),
  #  make_convolutional_layer(4, 4, 32, 64, 3, 2),
  #  make_activation_layer(RELU),
  #  make_maxpool_layer(4, 4, 64, 3, 2),
  #  make_connected_layer(256, 10),
  #  make_activation_layer(SOFTMAX)]

  #To TEST FOR Connected (Normal DL) Architecture:
  # l = [   make_connected_layer(32, 32, 3, 8, 3, 1),
  #         make_activation_layer(LEAKY RELU),
  #         make_connected_layer(256, 10),
  #         make_activation_layer(SOFTMAX)]

  return make_net(l)

print("loading data...")
train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels")
test  = load_image_classification_data("cifar/cifar.test",  "cifar/cifar.labels")
print("done")
print

print("making model...")
batch = 128
iters = 5000
rate = .01
momentum = .9
decay = .005

m = conv_net()
print("training...")
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_net(m, train))
print("test accuracy:     %f", accuracy_net(m, test))

# EXPERIMENTS:
# Total # of Operations in given Convolutional Architecture
#
# We know that we are only considering matrix operations, and there is only
# one matmul going forward in a layer which only occurs in the convolutional and connected layers.
#
# make_convolutional_layer(32, 32, 3, 8, 3, 1),
# (3 * 3 * 3) * (32 * 32) * 8 = 221184 
# make_convolutional_layer(16, 16, 8, 16, 3, 1),
# (8 * 3 * 3) * (16 * 16) * 16 = 294912 
# make_convolutional_layer(8, 8, 16, 32, 3, 1),
# (16 * 3 * 3) * (8 * 8) * 32 = 294912 
# make_convolutional_layer(4, 4, 32, 64, 3, 1),
# (32 * 3 * 3) * (4 * 4) * 64 = 294912 
# make_connected_layer(256, 10)
# This is just 256 * 10 * 1 = 2560
#
# Total = 1108480 operations

# How accurate is the fully connected network vs the convnet when they use similar number of operations?
# Why are you seeing these results? Speculate based on the information you've gathered and what you know about DL and ML.
# Your answer:
#


