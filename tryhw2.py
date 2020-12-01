from uwnet import *
def conv_net():
    l = [   make_convolutional_layer(32, 32, 3, 8, 3, 2),
           make_activation_layer(RELU),
           make_maxpool_layer(16, 16, 8, 3, 2),
           make_convolutional_layer(8, 8, 8, 16, 3, 1),
           make_activation_layer(RELU),
           make_maxpool_layer(8, 8, 16, 3, 2),
           make_convolutional_layer(4, 4, 16, 32, 3, 1),
           make_activation_layer(RELU),
           make_connected_layer(512, 10),
           make_activation_layer(SOFTMAX)]

    # l = [   make_convolutional_layer(32, 32, 3, 8, 3, 2), 
    #         make_activation_layer(RELU),
    #         make_convolutional_layer(16, 16, 8, 16, 3, 2), 
    #         make_activation_layer(RELU),
    #         make_convolutional_layer(8, 8, 16, 32, 3, 2), 
    #         make_activation_layer(RELU),
    #         make_convolutional_layer(4, 4, 32, 64, 3, 2), 
    #         make_activation_layer(RELU),
    #         make_connected_layer(256, 10),
    #         make_activation_layer(SOFTMAX)]
    return make_net(l)

def conv_batch_net():
    l = [   make_convolutional_layer(32, 32, 3, 8, 3, 2),
           make_batchnorm_layer(8),
           make_activation_layer(RELU),
           make_maxpool_layer(16, 16, 8, 3, 2),
           make_convolutional_layer(8, 8, 8, 16, 3, 1),
           make_batchnorm_layer(16),
           make_activation_layer(RELU),
           make_maxpool_layer(8, 8, 16, 3, 2),
           make_convolutional_layer(4, 4, 16, 32, 3, 1),
           make_batchnorm_layer(32),
           make_activation_layer(RELU),
           make_connected_layer(512, 10),
           make_activation_layer(SOFTMAX)]

    # l = [   make_convolutional_layer(32, 32, 3, 8, 3, 2), 
    #         make_batchnorm_layer(8),
    #         make_activation_layer(RELU),
    #         make_convolutional_layer(16, 16, 8, 16, 3, 2), 
    #         make_batchnorm_layer(16),
    #         make_activation_layer(RELU),
    #         make_convolutional_layer(8, 8, 16, 32, 3, 2), 
    #         make_batchnorm_layer(32),
    #         make_activation_layer(RELU),
    #         make_convolutional_layer(4, 4, 32, 64, 3, 2), 
    #         make_activation_layer(RELU),
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
iters = 500
rate = 0.005
momentum = .9
decay = .005

#m = conv_net()
m = conv_batch_net()

print("training...")
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_net(m, train))
print("test accuracy:     %f", accuracy_net(m, test))

# 7.6 Question: What do you notice about training the convnet with/without batch normalization? How does it affect convergence? How does it affect what magnitude of learning rate you can use? Write down any observations from your experiments:
# TODO: Your answer

#Learning rate 0.038
# Convolutional Training: ~0.23 
# Convolutional Testing: ~0.23
#
# Convolutional w/ BatchNorm Training: ~0.45
# Convolutional w/ BatchNorm: ~0.44

# Convolutional w/ Batch norm had a higher accuracy.

#With learning rate .1, our batch norm convolutional net had 
# Training Accuracy: ~0.365
# Testing Accuracy: ~0.368

#With learning rate .09, our batch norm convolutional net had 
# Training Accuracy: ~0.4
# Testing Accuracy: ~0.4

#With learning rate .08, our batch norm convolutional net had 
# Training Accuracy: ~0.424
# Testing Accuracy: ~0.427

#With learning rate .07, our batch norm convolutional net had 
# Training Accuracy: ~0.431
# Testing Accuracy: ~0.435

#With learning rate .06, our batch norm convolutional net had 
# Training Accuracy: ~0.447
# Testing Accuracy: ~0.446

#With learning rate .05, our batch norm convolutional net had 
# Training Accuracy: ~0.427
# Testing Accuracy: ~0.423

#With learning rate .04, our batch norm convolutional net had 
# Training Accuracy: ~0.429
# Testing Accuracy: ~0.425

#With learning rate .03, our batch norm convolutional net had 
# Training Accuracy: ~0.444
# Testing Accuracy: ~0.440

#With learning rate .02, our batch norm convolutional net had 
# Training Accuracy: ~0.447
# Testing Accuracy: ~0.448

#BEST RESULT##################################################
# With learning rate .01, our batch norm convolutional net had
# Training Accuracy: ~0.470 
# Testing Accuracy: ~0.463
##############################################################

#With learning rate .005, our batch norm convolutional net had 
# Training Accuracy: ~0.46
# Testing Accuracy: ~0.46