
import pickle
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import time
import glob
import matplotlib.image as mpimg
import cv2


dothetrain =1
############################################################################
####################              1. LOAD THE DATA       ###################
############################################################################

print("\n1. Loading Data")
training_file = "./traffic-signs-data/train.p"
validation_file = "./traffic-signs-data/valid.p"
testing_file = "./traffic-signs-data/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels'] # Providing the labels and the number of images for the train set
X_valid, y_valid = valid['features'], valid['labels'] # Providing the labels and the number of images for the valid set
X_test, y_test = test['features'], test['labels']     # Providing the labels and the number of images for the test set

############################################################################
####################        2. Dataset Exploration       ###################
############################################################################

### Giving an Overview of the used data

print("\n2. Dataset Exploration")
n_train = len(X_train)
n_valid = len(X_valid)
n_test = len(X_test)
image_shape = X_train[0].shape      #  Gives us the size of an image and so the dimension of the data -> LeNet needs 32x32x1
n_classes = len(np.unique(y_train))

#  Prinitng the size of each dataset
print("Number of training examples =", n_train)
print("Number of validation examples =", n_valid)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

### Visulazing the data by showing the image of 5 random data points -> Shows us that everything is working as we thougth i will
### Printing the image with the label -> The should match

# fig, axs = plt.subplots(1,5, figsize=(15, 6))
# fig.subplots_adjust(hspace = .01, wspace=.5)
# axs = axs.ravel()
# for i in range(5):
#     index = random.randint(0, len(X_train))
#     image = X_train[index]
#     axs[i].axis('off')
#     axs[i].imshow(image)
#     axs[i].set_title(y_train[index])

#plt.show()

### Visulazing the data by showing the histogram for each dataset


# hist, bins = np.histogram(y_train, bins=n_classes)
# width = 0.7 * (bins[1] - bins[0])
# center = (bins[:-1] + bins[1:]) / 2
# plt.bar(center, hist, align='center', width=width)
# plt.title("Train Dataset Sign Counts")
# #plt.show()
#
# hist1, bins1 = np.histogram(y_valid , bins=n_classes)
# width = 0.7 * (bins1[1] - bins1[0])
# center = (bins1[:-1] + bins1[1:]) / 2
# plt.bar(center, hist1, align='center', width=width)
# plt.title("Valid Dataset Sign Counts")
# #plt.show()
#
# hist2, bins2 = np.histogram(y_test, bins=n_classes)
# width = 0.7 * (bins2[1] - bins2[0])
# center = (bins2[:-1] + bins2[1:]) / 2
# plt.bar(center, hist2, align='center', width=width)
# plt.title("Test Dataset Sign Counts")
# #plt.show()


##################################################################################################
####################            3. Design, Train and Test Model                ###################
##################################################################################################

####################                   3.1 Preprocess                          ###################

print("\n3. NN Model: Preprocessing")
# 1. Preprocessing the data

# Turning the image to greyscale
X_train_rgb = X_train
X_train_gry = np.sum(X_train/3, axis=3, keepdims=True)

X_valid_rgb = X_valid
X_valid_gry = np.sum(X_valid/3, axis=3, keepdims=True)

X_test_rgb = X_test
X_test_gry = np.sum(X_test/3, axis=3, keepdims=True)


X_train = X_train_gry
X_test = X_test_gry
X_valid = X_valid_gry

print('RGB shape:', X_train_rgb.shape)
print('Grayscale shape:', X_train_gry.shape)
print('done')

######### Normalizing the data

print('Mean value of train dataset before normalizaion:', np.mean(X_train))
print('Mean value of valid dataset before normalizaion:', np.mean(X_valid))
print('Mean value of test dataset before normalizaion:', np.mean(X_test))

X_train = (X_train - 128)/128
X_valid = (X_valid - 128)/128
X_test = (X_test - 128)/128

print('\nMean value of train dataset before normalizaion:', np.mean(X_train))
print('Mean value of valid dataset before normalizaion:', np.mean(X_valid))
print('Mean value of test dataset before normalizaion:', np.mean(X_test))


#########  Shuffle the training dataset

X_train, y_train = shuffle(X_train, y_train)



####################                   3.2 Model Architecture                        ###################


# Defining the NN paramters, The EPOCH and BATCH_SIZE values affect the training speed and model accuracy.
EPOCHS = 70            # how many times the training data should be runned through the network, more epochs, more accuracy, longer time
BATCH_SIZE = 100        # how many Training images should be run through the NN at a time


def LeNet(x):

    #Hyperparamters: Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)

    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)

    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    fc0 = flatten(conv2)

    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b

    # SOLUTION: Activation.
    fc1 = tf.nn.relu(fc1)

    # Additional Dropout, preventing Overfitting
    fc1 = tf.nn.dropout(fc1, keep_prob)

    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b

    # SOLUTION: Activation.
    fc2 = tf.nn.relu(fc2)

    # Additional Dropout, preventing Overfitting
    fc2 = tf.nn.dropout(fc2, keep_prob)

    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 43.
    fc3_W = tf.Variable(tf.truncated_normal(shape=(84, 43), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits

####################                   3.3 Training Pipeline                       ###################

# Defining the variables for Tensorflow
x = tf.placeholder(tf.float32, (None, 32, 32, 1))       # Placeholder that will store the input badges, initilaize badge size to None, image dimension 32x32x2
y = tf.placeholder(tf.int32, (None))                    # Placeholder that will store labels, Labels are Integers
keep_prob = tf.placeholder(tf.float32)                  # probability to keep units
one_hot_y = tf.one_hot(y, 43)                           # One hot encoding the labels

# Setup the Training Pipeline
rate = 0.0007                                                                                 # Learning Rate of the NN, show how quickly to update the weights

logits = LeNet(x)                                                                            # Input Data will be passed in the LeNet Function to calculate the logits
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)     # Cross-Entropy: Compare the logits to the ground truth labels - Measurement how different the logits from the ground truth
loss_operation = tf.reduce_mean(cross_entropy)                                               # Averages the cross entropy
optimizer = tf.train.AdamOptimizer(learning_rate = rate)                                     # Gradient descent: minimizes the loss function
training_operation = optimizer.minimize(loss_operation)                                      # Run the minimizer to the optimizer: Backpropagation to reduce the trainings loss

####################                   3.4 Evaluation Pipeline                  ###################

# Evaluate how well the loss and accuracy of the model for a given dataset. How good is the Model
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))    # Measure if a given prediction is correct: comparing logit prediction to one hot encoded
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))    # Calculate the models overall accuracy by averaging the individual prediction accuracy
saver = tf.train.Saver()

# The evaluate function: Takes one dataset as an input, sets initial variables, batches the dataset and runs it trough the pipeline
def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

####################                   3.5 Training the Model                       ###################

print("\n4. NN Model: Train the Model")


if dothetrain == 1:
    with tf.Session() as sess:  # Create the Tensorflow Session
        sess.run(tf.global_variables_initializer())  # Initialize the variables
        num_examples = len(X_train)

        validation_accuracy_figure = []  # List for storing the data

        print("Training...")
        print()

        start = time.time()
        for i in range(EPOCHS):
            X_train, y_train = shuffle(X_train, y_train)  # We shuffle the data so it isnt biased by the images
            for offset in range(0, num_examples,
                                BATCH_SIZE):  # The training data is broken into batches and the training is done for each batch
                end = offset + BATCH_SIZE
                batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})

            validation_accuracy = evaluate(X_valid,
                                           y_valid)  # At the end of each epoch, we evaluate the model with the validation data
            validation_accuracy_figure.append(validation_accuracy)
            print("EPOCH {} ...".format(i + 1))
            print("Validation Accuracy = {:.3f}".format(validation_accuracy))
            print()

        saver.save(sess, './lenet')
        print("Model saved")
        end = time.time()
        print("Time for Training the model:", (end - start), "s")

        # plt.plot(validation_accuracy_figure)
        # plt.title("Validation Accuracy")
        # #plt.show()
else:
    test = 1

####################                   3.6  Test the Model                       ###################

print("\n5. NN Model: Test the Model with the Test Data")

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver2 = tf.train.import_meta_graph('./lenet.meta')
    saver2.restore(sess, "./lenet")
    test_accuracy = evaluate(X_test, y_test)
    print("Test Set Accuracy = {:.3f}".format(test_accuracy))



##############################################################################################
####################              4. Test the Data on new images           ###################
##############################################################################################

print("\n6. Test the Model with new images")


####################              4.1. Acquiering new images         ########################
my_images = []
my_labels = [3, 11, 1, 12, 38, 34, 25]

for i, img in enumerate(glob.glob('./new-traffic-signs/*x.png')):
    image = cv2.imread(img)
    my_images.append(image)

my_images = np.asarray(my_images)
my_images_gry = np.sum(my_images/3, axis=3, keepdims=True)                  # greyscaling new images
my_images_normalized = (my_images_gry - 128)/128                            # normalizing new images


####################              4.2. Accuracy of the new data        ########################

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver3 = tf.train.import_meta_graph('./lenet.meta')
    saver3.restore(sess, "./lenet")
    my_accuracy = evaluate(my_images_normalized, my_labels)
    print("New Images - Set Accuracy = {:.3f}".format(my_accuracy))


####################              4.3. Softmax Tests        ########################
softmax_logits = tf.nn.softmax(logits)
top_k = tf.nn.top_k(softmax_logits, k=3)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph('./lenet.meta')
    saver.restore(sess, "./lenet")
    my_softmax_logits = sess.run(softmax_logits, feed_dict={x: my_images_normalized, keep_prob: 1.0})
    my_top_k = sess.run(top_k, feed_dict={x: my_images_normalized, keep_prob: 1.0})

    fig, axs = plt.subplots(len(my_images), 4, figsize=(12, 14))
    fig.subplots_adjust(hspace=.4, wspace=.2)
    axs = axs.ravel()

    for i, image in enumerate(my_images):
        axs[4 * i].axis('off')
        axs[4 * i].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axs[4 * i].set_title('input')
        guess1 = my_top_k[1][i][0]
        index1 = np.argwhere(y_valid == guess1)[0]
        axs[4 * i + 1].axis('off')
        axs[4 * i + 1].imshow(X_valid[index1].squeeze(), cmap='gray')
        axs[4 * i + 1].set_title('top guess: {} ({:.0f}%)'.format(guess1, 100 * my_top_k[0][i][0]))
        guess2 = my_top_k[1][i][1]
        index2 = np.argwhere(y_valid == guess2)[0]
        axs[4 * i + 2].axis('off')
        axs[4 * i + 2].imshow(X_valid[index2].squeeze(), cmap='gray')
        axs[4 * i + 2].set_title('2nd guess: {} ({:.0f}%)'.format(guess2, 100 * my_top_k[0][i][1]))
        guess3 = my_top_k[1][i][2]
        index3 = np.argwhere(y_valid == guess3)[0]
        axs[4 * i + 3].axis('off')
        axs[4 * i + 3].imshow(X_valid[index3].squeeze(), cmap='gray')
        axs[4 * i + 3].set_title('3rd guess: {} ({:.0f}%)'.format(guess3, 100 * my_top_k[0][i][2]))
    plt.show()


############################################################################################
####################              5. Additional Visualization            ###################
############################################################################################

### Visualize your network's feature maps here.
### Feel free to use as many code cells as needed.

# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry

def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
    activation = tf_activation.eval(session=sess,feed_dict={x : image_input})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,15))
    for featuremap in range(featuremaps):
        plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")