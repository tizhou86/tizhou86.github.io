# TensorFlow: MNIST for experts

This scenario is the continuation of the MNIST for beginner one and shows how to use TensorFlow to build deep convolutional network. The training is performed on the MNIST dataset that is considered a Hello world for the deep learning examples. The content is based on the official TensorFlow tutorial.


##Training process
As mentioned in the MNIST for beginners tutorial, our deep learning process was defined by few steps:

Reading training/testing dataset (MNIST)
Defining the neural network architecture
Defining the loss function and optimiser method
Training the neural network based on data batches
Evaluating the performance over the test data
Here we cumulated almost every step from this list and will be working strictly on the neural network architecture design part.

The whole process was defined in the `training.py` file, and will be imported in the specific files including the code for different neural network architectures. The `dense.py` includes the simple one output layer network, which is the same as presented in the beginners tutorial. Later in the scenario, we will create a more complex model.

train.py
```
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def train_network(training_data, labels, output, keep_prob=tf.placeholder(tf.float32)):
    learning_rate = 1e-4
    steps_number = 1000
    batch_size = 100

    # Read data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # Define the loss function
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=output))

    # Training step
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    # Accuracy calculation
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Run the training
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    for i in range(steps_number):
        # Get the next batch
        input_batch, labels_batch = mnist.train.next_batch(batch_size)

        # Print the accuracy progress on the batch every 100 steps
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={training_data: input_batch, labels: labels_batch, keep_prob: 1.0})
            print("Step %d, training batch accuracy %g %%"%(i, train_accuracy*100))

        # Run the training step
        train_step.run(feed_dict={training_data: input_batch, labels: labels_batch, keep_prob: 0.5})

    print("The end of training!")

    # Evaluate on the test set
    test_accuracy = accuracy.eval(feed_dict={training_data: mnist.test.images, labels: mnist.test.labels, keep_prob: 1.0})
    print("Test accuracy: %g %%"%(test_accuracy*100))

```
dense.py
```
import tensorflow as tf

image_size = 28
labels_size = 10

# Define placeholders
training_data = tf.placeholder(tf.float32, [None, image_size*image_size])
labels = tf.placeholder(tf.float32, [None, labels_size])

# Variables to be tuned
W = tf.Variable(tf.truncated_normal([image_size*image_size, labels_size], stddev=0.1))
b = tf.Variable(tf.constant(0.1, shape=[labels_size]))

# Build the network (only output layer)
output = tf.matmul(training_data, W) + b

# Train & test the network
import training
training.train_network(training_data, labels, output)

```

You can run the network training by using the following command:
```
  > python dense.py
Successfully downloaded train-images-idx3-ubyte.gz 9912422 bytes.
Extracting MNIST_data/train-images-idx3-ubyte.gz
Successfully downloaded train-labels-idx1-ubyte.gz 28881 bytes.
Extracting MNIST_data/train-labels-idx1-ubyte.gz
Successfully downloaded t10k-images-idx3-ubyte.gz 1648877 bytes.
Extracting MNIST_data/t10k-images-idx3-ubyte.gz
Successfully downloaded t10k-labels-idx1-ubyte.gz 4542 bytes.
Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
Step 0, training batch accuracy 3 %
Step 100, training batch accuracy 20 %
Step 200, training batch accuracy 48 %
Step 300, training batch accuracy 56 %
Step 400, training batch accuracy 60 %
Step 500, training batch accuracy 59 %
Step 600, training batch accuracy 68 %
Step 700, training batch accuracy 77 %
Step 800, training batch accuracy 83 %
Step 900, training batch accuracy 76 %
The end of training!
Test accuracy: 81.38 %
```

##Hidden layer
The default architecture consists of only one output layer. This influences both performance and accuracy. In this step, we will add another dense layer to the network. The whole code can bee seen in the `hidden.py` file.

![图片](http://bos.nj.bpc.baidu.com/v1/agroup/19f8fbeecaece337b02adf7ea5a5ebf03d93cec6)

As in the case of one output layer, the code should start with defining the placeholders, which are representing flattened digits' images and the corresponding labels.
```
training_data = tf.placeholder(tf.float32, [None, image_size*image_size])
labels = tf.placeholder(tf.float32, [None, labels_size])
```
Then we define the hidden layer as the dense one. We chose to use 1024 neurons and `reLU`) as the activation function. Because it will be another dense layer, we need to define the weights and biases variables.
```
W_h = tf.Variable(tf.truncated_normal([image_size*image_size, hidden_size], stddev=0.1))
b_h = tf.Variable(tf.constant(0.1, shape=[hidden_size]))
```
TensorFlow provides the tf.nn.relu function that will be applied after performing the matrices multiplication.
```
hidden = tf.nn.relu(tf.matmul(training_data, W_h) + b_h)
```
As a finishing touch, we connect hidden layer with the output one and return required objects. Notice that we changed the dimension for the weights variable to fit the hidden layer instead of the input one.
```
W = tf.Variable(tf.truncated_normal([hidden_size, labels_size], stddev=0.1))
b = tf.Variable(tf.constant(0.1, shape=[labels_size]))
```
```
output = tf.matmul(hidden, W) + b
```
hidden.py

```
import tensorflow as tf

image_size = 28
labels_size = 10
hidden_size = 1024

# Define placeholders
training_data = tf.placeholder(tf.float32, [None, image_size*image_size])
labels = tf.placeholder(tf.float32, [None, labels_size])

# Variables for the hidden layer
W_h = tf.Variable(tf.truncated_normal([image_size*image_size, hidden_size], stddev=0.1))
b_h = tf.Variable(tf.constant(0.1, shape=[hidden_size]))

# Hidden layer with reLU activation function
hidden = tf.nn.relu(tf.matmul(training_data, W_h) + b_h)

# Variables for the output layer
W = tf.Variable(tf.truncated_normal([hidden_size, labels_size], stddev=0.1))
b = tf.Variable(tf.constant(0.1, shape=[labels_size]))

# Connect hidden to the output layer
output = tf.matmul(hidden, W) + b

# Train & test the network
import training
training.train_network(training_data, labels, output)

```

You can run the whole thing using the following command:
```
  > python hidden.py
Successfully downloaded train-images-idx3-ubyte.gz 9912422 bytes.
Extracting MNIST_data/train-images-idx3-ubyte.gz
Successfully downloaded train-labels-idx1-ubyte.gz 28881 bytes.
Extracting MNIST_data/train-labels-idx1-ubyte.gz
Successfully downloaded t10k-images-idx3-ubyte.gz 1648877 bytes.
Extracting MNIST_data/t10k-images-idx3-ubyte.gz
Successfully downloaded t10k-labels-idx1-ubyte.gz 4542 bytes.
Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
Step 0, training batch accuracy 8 %
Step 100, training batch accuracy 83 %
Step 200, training batch accuracy 92 %
Step 300, training batch accuracy 87 %
Step 400, training batch accuracy 95 %
Step 500, training batch accuracy 93 %
Step 600, training batch accuracy 91 %
Step 700, training batch accuracy 90 %
Step 800, training batch accuracy 96 %
Step 900, training batch accuracy 95 %
The end of training!
Test accuracy: 94.15 %
```

##Convolutional layer
The next two layers we're going to add are the integral parts of convolutional networks. They work differently than the dense ones and perform especially well with 2- and more dimensions input. 

![图片](http://bos.nj.bpc.baidu.com/v1/agroup/7a7e08bf716ffeac9569a093797903ace2b9bed2)

The parameters of the convolutional layer are the size of the convolution window and the strides. Padding set as 'SAME' indicates that the resulting layer is of the same size. After this step, we apply max pooling. We will build two convolutional layers, connect it to the dense hidden layer. The resulting architecture can be visualised as follows:

![Convolutional network](http://bos.nj.bpc.baidu.com/v1/agroup/7d5ed423b6854a2a470706dab9ccba630fcc3491)

The code for the whole network can be found in the `convolutional.py` and we will walk you through it now. In the previous example placeholders are representing flattened digits' images and the corresponding labels. Although, a convolutional layer can work with higher dimensional data, so we need to reshape the images.
```
training_data = tf.placeholder(tf.float32, [None, image_size*image_size])
training_images = tf.reshape(training_data, [-1, image_size, image_size, 1])
labels = tf.placeholder(tf.float32, [None, labels_size])
```
Next step is to set up the variables for the first convolutional layer. Then we initiate the convolutional and max polling phases. As you can see we use the variety of tf.nn functions like relu, conv2d or max_pool. This layer reads the reshaped images directly from the input data.
```
W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
```
```
conv1 = tf.nn.relu(tf.nn.conv2d(training_images, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
```
The second layer is analogical and has been defined below. Notice that as in input it takes the result of the max polling from the previous step.
```
W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
```
```
conv2 = tf.nn.relu(tf.nn.conv2d(pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
```
The last thing left is to connect it to the next layer which is a dense hidden one. Dense layers don't work with the dimensions of the convolution, so we need to flatten the result from the convolution phase.
```
pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
```
The next step will show how to connect it to the hidden dense layer.

convolutional.py

```
import tensorflow as tf

image_size = 28
labels_size = 10
hidden_size = 1024

# Define placeholders
training_data = tf.placeholder(tf.float32, [None, image_size*image_size])
training_images = tf.reshape(training_data, [-1, image_size, image_size, 1])

labels = tf.placeholder(tf.float32, [None, labels_size])

# 1st convolutional layer variables
W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))

# 1st convolution & max pooling
conv1 = tf.nn.relu(tf.nn.conv2d(training_images, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 2nd convolutional layer variables
W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))

# 2nd convolution & max pooling
conv2 = tf.nn.relu(tf.nn.conv2d(pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Flatten the 2nd convolution layer
pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

#Variables for the hidden dense layer
W_h = tf.Variable(tf.truncated_normal([7 * 7 * 64, hidden_size], stddev=0.1))
b_h = tf.Variable(tf.constant(0.1, shape=[hidden_size]))

# Hidden layer with reLU activation function
hidden = tf.nn.relu(tf.matmul(pool2_flat, W_h) + b_h)

# Dropout
keep_prob = tf.placeholder(tf.float32)
hidden_drop = tf.nn.dropout(hidden, keep_prob)

# Variables to be tuned
W = tf.Variable(tf.truncated_normal([hidden_size, labels_size], stddev=0.1))
b = tf.Variable(tf.constant(0.1, shape=[labels_size]))

# Connect hidden to the output layer
output = tf.matmul(hidden_drop, W) + b

# Train & test the network
import training
training.train_network(training_data, labels, output, keep_prob)

```

##Dropout
In the last step, we created two convolutional layers and flattened the results. Now it's time to connect it to the dense hidden layer. This is analogical to what we've already seen in the second step of the tutorial with the difference that this time we use pool2_flat instead of the images as input.
```
W_h = tf.Variable(tf.truncated_normal([7 * 7 * 64, hidden_size], stddev=0.1))
b_h = tf.Variable(tf.constant(0.1, shape=[hidden_size]))
```
```
hidden = tf.nn.relu(tf.matmul(pool2_flat, W_h) + b_h)
```
We could now just connect the hidden dense layer to the output, but we would like to do one more thing - apply the dropout. Dropout is the technique that is used to avoid overfitting by not using some neurons when training the network.

![Dropout](http://bos.nj.bpc.baidu.com/v1/agroup/e800bf0f2bf638fa743d97b53a2bfb0af7b0d60f)

Dropout works in a way that individual nodes are either "shut down" or kept with some explicit probability. It is used in the training phase, so remember you need to turn it off when evaluating your network. TensorFlow is allowing you to use the dropout function to implement it.Dropout works in a way that individual nodes are either "shut down" or kept with some explicit probability. It is used in the training phase, so remember you need to turn it off when evaluating your network. TensorFlow is allowing you to use the dropout function to implement it.

![图片](http://bos.nj.bpc.baidu.com/v1/agroup/f32745766460b045e5ee333623b713f8adefba80)



It's very important to dropout the neurons only in the training phase, but not when evaluating the model. This is why define an additional placeholder to keep the dropout probability. Then we use tf.nn.dropout function.

```
keep_prob = tf.placeholder(tf.float32)
hidden_drop = tf.nn.dropout(hidden, keep_prob)
```
The last step is to connect it to the output layer.
```
W = tf.Variable(tf.truncated_normal([hidden_size, labels_size], stddev=0.1))
b = tf.Variable(tf.constant(0.1, shape=[labels_size]))
```
```
output = tf.matmul(hidden_drop, W) + b
```
You can run the code with the command:
```
  > python convolutional.py
Successfully downloaded train-images-idx3-ubyte.gz 9912422 bytes.
Extracting MNIST_data/train-images-idx3-ubyte.gz
Successfully downloaded train-labels-idx1-ubyte.gz 28881 bytes.
Extracting MNIST_data/train-labels-idx1-ubyte.gz
Successfully downloaded t10k-images-idx3-ubyte.gz 1648877 bytes.
Extracting MNIST_data/t10k-images-idx3-ubyte.gz
Successfully downloaded t10k-labels-idx1-ubyte.gz 4542 bytes.
Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
Step 0, training batch accuracy 11 %
Step 100, training batch accuracy 78 %
Step 200, training batch accuracy 90 %
Step 300, training batch accuracy 93 %
Step 400, training batch accuracy 91 %
Step 500, training batch accuracy 93 %
Step 600, training batch accuracy 96 %
Step 700, training batch accuracy 94 %
Step 800, training batch accuracy 95 %
Step 900, training batch accuracy 97 %
The end of training!
Test accuracy: 97.11 %
```
Notice that the complexity of the network is influencing the speed of training, but also improving the accuracy. Try to adjust the result by changing the training parameters.

