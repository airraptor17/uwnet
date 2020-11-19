from uwnet import *
def conv_net():
    #l = [   make_convolutional_layer(32, 32, 3, 8, 3, 2),
    #        make_activation_layer(RELU),
    #        make_maxpool_layer(16, 16, 8, 3, 2),
    #        make_convolutional_layer(8, 8, 8, 16, 3, 1),
    #        make_activation_layer(RELU),
    #        make_maxpool_layer(8, 8, 16, 3, 2),
    #        make_convolutional_layer(4, 4, 16, 32, 3, 1),
    #        make_activation_layer(RELU),
    #        make_connected_layer(512, 10),
    #        make_activation_layer(SOFTMAX)]

    l = [   make_convolutional_layer(32, 32, 3, 8, 3, 2), 
            make_activation_layer(RELU),
            make_convolutional_layer(16, 16, 8, 16, 3, 2), 
            make_activation_layer(RELU),
            make_convolutional_layer(8, 8, 16, 32, 3, 2), 
            make_activation_layer(RELU),
            make_convolutional_layer(4, 4, 32, 64, 3, 2), 
            make_activation_layer(RELU),
            make_connected_layer(256, 10),
            make_activation_layer(SOFTMAX)]
    return make_net(l)

def conv_batch_net():
    #l = [   make_convolutional_layer(32, 32, 3, 8, 3, 2),
    #        make_batchnorm_layer(8),
    #        make_activation_layer(RELU),
    #        make_maxpool_layer(16, 16, 8, 3, 2),
    #        make_convolutional_layer(8, 8, 8, 16, 3, 1),
    #        make_batchnorm_layer(16),
    #        make_activation_layer(RELU),
    #        make_maxpool_layer(8, 8, 16, 3, 2),
    #        make_convolutional_layer(4, 4, 16, 32, 3, 1),
    #        make_batchnorm_layer(32),
    #        make_activation_layer(RELU),
    #        make_connected_layer(512, 10),
    #        make_activation_layer(SOFTMAX)]

    l = [   make_convolutional_layer(32, 32, 3, 8, 3, 2), 
            make_batchnorm_layer(8),
            make_activation_layer(RELU),
            make_convolutional_layer(16, 16, 8, 16, 3, 2), 
            make_batchnorm_layer(16),
            make_activation_layer(RELU),
            make_convolutional_layer(8, 8, 16, 32, 3, 2), 
            make_batchnorm_layer(32),
            make_activation_layer(RELU),
            make_convolutional_layer(4, 4, 32, 64, 3, 2), 
            make_activation_layer(RELU),
            make_connected_layer(256, 10),
            make_activation_layer(SOFTMAX)]
    return make_net(l)


print("loading data...")
train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels")
test  = load_image_classification_data("cifar/cifar.test",  "cifar/cifar.labels")
print("done")
print

print("making model...")
batch = 128
iters = 500
rate = 0.03
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

# Convolutional Training: ~0.42
# Convolutional Testing: ~0.41
#
# Convolutional w/ BatchNorm Training: ~0.52
# Convolutional w/ BatchNorm: ~0.51

#Convolutional w/ Batch norm had a higher accuracy.

#With learning rate .1, our batch norm convolutional net had 
# Training Accuracy: ~0.538
# Testing Accuracy: ~0.524

#With learning rate .07, our batch norm convolutional net had 
# Training Accuracy: ~0.561
# Testing Accuracy: ~0.538
#Which was the best results we got.