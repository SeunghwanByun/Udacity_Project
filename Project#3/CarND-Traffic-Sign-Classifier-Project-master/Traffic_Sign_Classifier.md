
# Self-Driving Car Engineer Nanodegree

## Deep Learning

## Project: Build a Traffic Sign Recognition Classifier

In this notebook, a template is provided for you to implement your functionality in stages, which is required to successfully complete this project. If additional code is required that cannot be included in the notebook, be sure that the Python code is successfully imported and included in your submission if necessary. 

> **Note**: Once you have completed all of the code implementations, you need to finalize your work by exporting the iPython Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to  \n",
    "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission. 

In addition to implementing code, there is a writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) that can be used to guide the writing process. Completing the code template and writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/481/view) for this project.

The [rubric](https://review.udacity.com/#!/rubrics/481/view) contains "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. The stand out suggestions are optional. If you decide to pursue the "stand out suggestions", you can include the code in this Ipython notebook and also discuss the results in the writeup file.


>**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

---
## Step 0: Load The Data


```python
# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file = "../data/train.p"
validation_file= "../data/valid.p"
testing_file = "../data/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']
```

---

## Step 1: Dataset Summary & Exploration

The pickled data is a dictionary with 4 key/value pairs:

- `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
- `'labels'` is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
- `'sizes'` is a list containing tuples, (width, height) representing the original width and height the image.
- `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. **THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES**

Complete the basic data summary below. Use python, numpy and/or pandas methods to calculate the data summary rather than hard coding the results. For example, the [pandas shape method](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.shape.html) might be useful for calculating some of the summary results. 

### Provide a Basic Summary of the Data Set Using Python, Numpy and/or Pandas


```python
### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results
import pandas as pd

data = pd.read_csv("signnames.csv", skiprows=[0], usecols=[0, 1], names=['Classid', 'SignName'])

# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of validation examples
n_validation = len(X_valid)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(data['Classid'])

print("Number of training examples =", n_train)
print("Number of validation examples =", n_validation)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)
```

    Number of training examples = 34799
    Number of validation examples = 4410
    Number of testing examples = 12630
    Image data shape = (32, 32, 3)
    Number of classes = 43


### Include an exploratory visualization of the dataset

Visualize the German Traffic Signs Dataset using the pickled file(s). This is open ended, suggestions include: plotting traffic sign images, plotting the count of each sign, etc. 

The [Matplotlib](http://matplotlib.org/) [examples](http://matplotlib.org/examples/index.html) and [gallery](http://matplotlib.org/gallery.html) pages are a great resource for doing visualizations in Python.

**NOTE:** It's recommended you start with something simple first. If you wish to do more, come back to it after you've completed the rest of the sections. It can be interesting to look at the distribution of classes in the training, validation and test set. Is the distribution the same? Are there more examples of some classes than others?


```python
### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.
%matplotlib inline

import numpy as np

number_train = []
number_validation = []
number_test = []

for class_n in range(n_classes):
    num_Each_class_train = np.where(y_train == class_n)
    num_Each_class_validation = np.where(y_valid == class_n)
    num_Each_class_test = np.where(y_test == class_n)
    
    number_train.append(len(num_Each_class_train[0]))
    number_validation.append(len(num_Each_class_validation[0]))
    number_test.append(len(num_Each_class_test[0]))


plt.bar(range(0, 43), number_train)
plt.xlabel('Classes')
plt.ylabel('Number of Samples')
plt.title('Train Data Visualization')
plt.show()

plt.bar(range(0, 43), number_validation)
plt.xlabel('Classes')
plt.ylabel('Number of Samples')
plt.title('Validation Data Visualization')
plt.show()

plt.bar(range(0, 43), number_test)
plt.xlabel('Classes')
plt.ylabel('Number of Samples')
plt.title('Test Data Visualization')
plt.show()
```


![png](output_8_0.png)



![png](output_8_1.png)



![png](output_8_2.png)


----

## Step 2: Design and Test a Model Architecture

Design and implement a deep learning model that learns to recognize traffic signs. Train and test your model on the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

The LeNet-5 implementation shown in the [classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) at the end of the CNN lesson is a solid starting point. You'll have to change the number of classes and possibly the preprocessing, but aside from that it's plug and play! 

With the LeNet-5 solution from the lecture, you should expect a validation set accuracy of about 0.89. To meet specifications, the validation set accuracy will need to be at least 0.93. It is possible to get an even higher accuracy, but 0.93 is the minimum for a successful project submission. 

There are various aspects to consider when thinking about this problem:

- Neural network architecture (is the network over or underfitting?)
- Play around preprocessing techniques (normalization, rgb to grayscale, etc)
- Number of examples per label (some have more than others).
- Generate fake data.

Here is an example of a [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). It's not required to be familiar with the approach used in the paper but, it's good practice to try to read papers like these.

### Pre-process the Data Set (normalization, grayscale, etc.)

Minimally, the image data should be normalized so that the data has mean zero and equal variance. For image data, `(pixel - 128)/ 128` is a quick way to approximately normalize the data and can be used in this project. 

Other pre-processing steps are optional. You can try different techniques to see if it improves performance. 

Use the code cell (or multiple code cells, if necessary) to implement the first step of your project.


```python
### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.

from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train)
```

### Model Architecture


```python
### Define your architecture here.
### Feel free to use as many code cells as needed.
import tensorflow as tf

EPOCHS = 50
BATCH_SIZE = 128
```


```python
from tensorflow.contrib.layers import flatten
import numpy as np

# First we make the architecture of cnn LeNet.
# And modify later.
def LeNet(x):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # Layer 1: Convolutional. Input = 32x32x3, Output = 28x28x6
    W1 = tf.Variable(tf.truncated_normal(shape=(5,5,3,6), mean=mu, stddev=sigma))
    b1 = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, W1, strides=[1,1,1,1], padding='VALID') + b1
    conv1 = tf.nn.relu(conv1)
    
    # Layer 2: Convolutional. Input = 28x28x6, Output = 24x24x10
    W2 = tf.Variable(tf.truncated_normal(shape=(5,5,6,10), mean=mu, stddev=sigma))
    b2 = tf.Variable(tf.zeros(10))
    conv2 = tf.nn.conv2d(conv1, W2, strides=[1,1,1,1], padding='VALID') + b2
    conv2 = tf.nn.relu(conv2)
    
    # Layer 3: Convolutional. Input = 24x24x10, Output = 20x20x16
    W3 = tf.Variable(tf.truncated_normal(shape=(5, 5, 10, 16), mean=mu, stddev=sigma))
    b3 = tf.Variable(tf.zeros(16))
    conv3 = tf.nn.conv2d(conv2, W3, strides=[1,1,1,1], padding='VALID') + b3
    conv3 = tf.nn.relu(conv3)
    
    # Layer 4: Convolutional. Input = 20x20x16, Output = 16x16x16
    W4 = tf.Variable(tf.truncated_normal(shape=(5, 5, 16, 32), mean=mu, stddev=sigma))
    b4 = tf.Variable(tf.zeros(32))
    conv4 = tf.nn.conv2d(conv3, W4, strides=[1,1,1,1], padding='VALID') + b4
    conv4 = tf.nn.relu(conv4)
    
    # Layer 5: Convolutional. Input = 16x16x32, Output = 12x12x64
    W5 = tf.Variable(tf.truncated_normal(shape=(5, 5, 32, 64), mean=mu, stddev=sigma))
    b5 = tf.Variable(tf.zeros(64))
    conv5 = tf.nn.conv2d(conv4, W5, strides=[1,1,1,1], padding='VALID') + b5
    conv5 = tf.nn.relu(conv5)
    
    # Layer 6: Convolutional. Input = 12x12x64, Output = 8x8x128
    W6 = tf.Variable(tf.truncated_normal(shape=(5, 5, 64, 128), mean=mu, stddev=sigma))
    b6 = tf.Variable(tf.zeros(128))
    conv6 = tf.nn.conv2d(conv5, W6, strides=[1,1,1,1], padding='VALID') + b6
    conv6 = tf.nn.relu(conv6)
    
    # Output = 4x4x128
    conv6 = tf.nn.max_pool(conv6, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    
    
    # Flatten. Input = 6x6x64.
    fc = flatten(conv6)
    
    # Layer 3: Fully Connected. Input = 4096, Output = 120.
    fc_W1 = tf.Variable(tf.truncated_normal(shape=(2048, 120), mean=mu, stddev=sigma))
    fc_b1 = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc, fc_W1) + fc_b1
    fc1 = tf.nn.relu(fc1)
    
    # Layer 4: Fully Connected. Input = 120, Ouput = 84.
    fc_W2 = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
    fc_b2 = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1, fc_W2) + fc_b2
    fc2 = tf.nn.relu(fc2)
    
    # Layer 5: Fully Connected. Input = 84. Output = n_classes.
    fc_W3 = tf.Variable(tf.truncated_normal(shape=(84, n_classes), mean=mu, stddev=sigma))
    fc_b3 = tf.Variable(tf.zeros(n_classes))
    logits = tf.matmul(fc2, fc_W3) + fc_b3
    
    return logits
```

### Train, Validate and Test the Model

A validation set can be used to assess how well the model is performing. A low accuracy on the training and validation
sets imply underfitting. A high accuracy on the training set but low accuracy on the validation set implies overfitting.


```python
### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.
x = tf.placeholder(tf.float32, (None,32,32,3))
y = tf.placeholder(tf.int32, (None))

one_hot_y = tf.one_hot(y, n_classes)
```


```python
rate = 0.001

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=rate)
training_operation = optimizer.minimize(loss_operation)
```


```python
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    total_cost = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy, cost = sess.run([accuracy_operation, loss_operation], feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
        total_cost += (cost * len(batch_x))
        
    return total_accuracy / num_examples, total_cost / num_examples
```


```python
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    history_cost = []
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        validation_accuracy, cost_accuracy = evaluate(X_valid, y_valid)
        history_cost.append(cost_accuracy)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './lenet')
    print("Model saved")
    
    plt.plot(range(0, EPOCHS), history_cost)
    plt.xlabel('Epochs')
    plt.ylabel('Cost Avg')
    plt.title('History of Cost reducing')
    plt.show()
```

    Training...
    
    EPOCH 1 ...
    Validation Accuracy = 0.676
    
    EPOCH 2 ...
    Validation Accuracy = 0.815
    
    EPOCH 3 ...
    Validation Accuracy = 0.859
    
    EPOCH 4 ...
    Validation Accuracy = 0.880
    
    EPOCH 5 ...
    Validation Accuracy = 0.877
    
    EPOCH 6 ...
    Validation Accuracy = 0.904
    
    EPOCH 7 ...
    Validation Accuracy = 0.919
    
    EPOCH 8 ...
    Validation Accuracy = 0.916
    
    EPOCH 9 ...
    Validation Accuracy = 0.916
    
    EPOCH 10 ...
    Validation Accuracy = 0.902
    
    EPOCH 11 ...
    Validation Accuracy = 0.922
    
    EPOCH 12 ...
    Validation Accuracy = 0.923
    
    EPOCH 13 ...
    Validation Accuracy = 0.926
    
    EPOCH 14 ...
    Validation Accuracy = 0.906
    
    EPOCH 15 ...
    Validation Accuracy = 0.946
    
    EPOCH 16 ...
    Validation Accuracy = 0.939
    
    EPOCH 17 ...
    Validation Accuracy = 0.935
    
    EPOCH 18 ...
    Validation Accuracy = 0.944
    
    EPOCH 19 ...
    Validation Accuracy = 0.946
    
    EPOCH 20 ...
    Validation Accuracy = 0.939
    
    EPOCH 21 ...
    Validation Accuracy = 0.907
    
    EPOCH 22 ...
    Validation Accuracy = 0.952
    
    EPOCH 23 ...
    Validation Accuracy = 0.948
    
    EPOCH 24 ...
    Validation Accuracy = 0.937
    
    EPOCH 25 ...
    Validation Accuracy = 0.925
    
    EPOCH 26 ...
    Validation Accuracy = 0.939
    
    EPOCH 27 ...
    Validation Accuracy = 0.939
    
    EPOCH 28 ...
    Validation Accuracy = 0.951
    
    EPOCH 29 ...
    Validation Accuracy = 0.945
    
    EPOCH 30 ...
    Validation Accuracy = 0.958
    
    EPOCH 31 ...
    Validation Accuracy = 0.952
    
    EPOCH 32 ...
    Validation Accuracy = 0.927
    
    EPOCH 33 ...
    Validation Accuracy = 0.950
    
    EPOCH 34 ...
    Validation Accuracy = 0.944
    
    EPOCH 35 ...
    Validation Accuracy = 0.947
    
    EPOCH 36 ...
    Validation Accuracy = 0.955
    
    EPOCH 37 ...
    Validation Accuracy = 0.965
    
    EPOCH 38 ...
    Validation Accuracy = 0.963
    
    EPOCH 39 ...
    Validation Accuracy = 0.945
    
    EPOCH 40 ...
    Validation Accuracy = 0.962
    
    EPOCH 41 ...
    Validation Accuracy = 0.965
    
    EPOCH 42 ...
    Validation Accuracy = 0.966
    
    EPOCH 43 ...
    Validation Accuracy = 0.966
    
    EPOCH 44 ...
    Validation Accuracy = 0.959
    
    EPOCH 45 ...
    Validation Accuracy = 0.959
    
    EPOCH 46 ...
    Validation Accuracy = 0.966
    
    EPOCH 47 ...
    Validation Accuracy = 0.944
    
    EPOCH 48 ...
    Validation Accuracy = 0.968
    
    EPOCH 49 ...
    Validation Accuracy = 0.964
    
    EPOCH 50 ...
    Validation Accuracy = 0.964
    
    Model saved



![png](output_21_1.png)



```python
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    
    test_accuracy, test_cost = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
```

    INFO:tensorflow:Restoring parameters from ./lenet
    Test Accuracy = 0.941


---

## Step 3: Test a Model on New Images

To give yourself more insight into how your model is working, download at least five pictures of German traffic signs from the web and use your model to predict the traffic sign type.

You may find `signnames.csv` useful as it contains mappings from the class id (integer) to the actual sign name.

### Load and Output the Images


```python
### Load the images and plot them here.
### Feel free to use as many code cells as needed.
import os
import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import matplotlib.image as mpimg
import cv2 as cv

images = os.listdir('new_traffic_sign')
path = './new_traffic_sign/'

new_image_test = []
new_image_labels = []

for img_name in images:
    img = cv.imread(path + img_name)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (32, 32), interpolation=cv.INTER_CUBIC)
    plt.imshow(img)
    plt.show()
    label = int(img_name[-6:-4])
    
    new_image_test.append(img)
    new_image_labels.append(label)    

```


![png](output_25_0.png)



![png](output_25_1.png)



![png](output_25_2.png)



![png](output_25_3.png)



![png](output_25_4.png)


### Predict the Sign Type for Each Image


```python
### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    new_image_accuracy, new_image_cost = evaluate(new_image_test, new_image_labels)
    print("Test Accuracy = {:.3f}".format(new_image_accuracy))
```

    INFO:tensorflow:Restoring parameters from ./lenet
    Test Accuracy = 1.000


### Analyze Performance


```python
### Calculate the accuracy for these 5 new images. 
### For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.
```

### Output Top 5 Softmax Probabilities For Each Image Found on the Web

For each of the new images, print out the model's softmax probabilities to show the **certainty** of the model's predictions (limit the output to the top 5 probabilities for each image). [`tf.nn.top_k`](https://www.tensorflow.org/versions/r0.12/api_docs/python/nn.html#top_k) could prove helpful here. 

The example below demonstrates how tf.nn.top_k can be used to find the top k predictions for each image.

`tf.nn.top_k` will return the values and indices (class ids) of the top k predictions. So if k=3, for each sign, it'll return the 3 largest probabilities (out of a possible 43) and the correspoding class ids.

Take this numpy array as an example. The values in the array represent predictions. The array contains softmax probabilities for five candidate images with six possible classes. `tf.nn.top_k` is used to choose the three classes with the highest probability:

```
# (5, 6) array
a = np.array([[ 0.24879643,  0.07032244,  0.12641572,  0.34763842,  0.07893497,
         0.12789202],
       [ 0.28086119,  0.27569815,  0.08594638,  0.0178669 ,  0.18063401,
         0.15899337],
       [ 0.26076848,  0.23664738,  0.08020603,  0.07001922,  0.1134371 ,
         0.23892179],
       [ 0.11943333,  0.29198961,  0.02605103,  0.26234032,  0.1351348 ,
         0.16505091],
       [ 0.09561176,  0.34396535,  0.0643941 ,  0.16240774,  0.24206137,
         0.09155967]])
```

Running it through `sess.run(tf.nn.top_k(tf.constant(a), k=3))` produces:

```
TopKV2(values=array([[ 0.34763842,  0.24879643,  0.12789202],
       [ 0.28086119,  0.27569815,  0.18063401],
       [ 0.26076848,  0.23892179,  0.23664738],
       [ 0.29198961,  0.26234032,  0.16505091],
       [ 0.34396535,  0.24206137,  0.16240774]]), indices=array([[3, 0, 5],
       [0, 1, 4],
       [0, 5, 1],
       [1, 3, 5],
       [1, 4, 3]], dtype=int32))
```

Looking just at the first row we get `[ 0.34763842,  0.24879643,  0.12789202]`, you can confirm these are the 3 largest probabilities in `a`. You'll also notice `[3, 0, 5]` are the corresponding indices.


```python
### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web. 
### Feel free to use as many code cells as needed.

softmax_logits = tf.nn.softmax(logits)
top_accuracy = tf.nn.top_k(softmax_logits, k=5)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    logits_, top_ = sess.run([softmax_logits, top_accuracy], feed_dict={x: new_image_test})
    print(logits_)
    print(top_)
```

    INFO:tensorflow:Restoring parameters from ./lenet
    [[1.21670347e-28 1.41758431e-13 7.07446657e-10 1.00000000e+00
      2.79847558e-24 2.92800215e-14 8.16344109e-24 1.35775345e-16
      1.26945144e-18 2.03602120e-17 2.27589193e-12 5.52216904e-15
      2.75285708e-20 5.89468928e-22 6.03882907e-15 2.97426024e-17
      1.37463009e-21 2.90098840e-31 1.92397424e-21 2.77564590e-19
      5.86329390e-29 4.28351937e-11 2.86904129e-30 7.08631879e-19
      3.96766333e-23 1.07523304e-13 9.57235325e-18 4.67230677e-22
      1.51474042e-17 2.36913404e-14 3.87725428e-19 1.72424734e-18
      8.86063981e-20 6.02743351e-16 3.20237679e-14 6.09216779e-13
      3.71183280e-33 4.64241547e-33 1.54334621e-14 2.98922725e-18
      2.08323882e-17 1.07984231e-18 3.18045288e-24]
     [6.07274717e-11 6.96645088e-07 1.05736353e-09 2.35112369e-04
      5.96453963e-12 5.06933702e-07 2.30445884e-14 8.96268086e-13
      6.85213331e-10 2.07765311e-06 6.70394229e-07 1.42903655e-05
      1.15818821e-08 1.84480911e-08 4.31885213e-13 9.49159062e-09
      1.24840598e-11 1.82108315e-10 2.97687963e-09 5.85863336e-06
      9.58356172e-10 4.86577512e-04 1.69490804e-08 2.43123566e-08
      3.17848436e-09 3.32113304e-05 4.47529338e-08 5.07871689e-10
      9.24411097e-07 2.16725098e-07 6.18342199e-07 9.99205768e-01
      7.19669376e-12 8.60608651e-09 9.74773755e-08 7.11601231e-08
      2.13871194e-14 7.70578317e-13 1.30875151e-05 1.30719267e-08
      6.66746214e-10 1.95884297e-16 2.46199539e-17]
     [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
      0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
      0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
      0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
      0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
      0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
      0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
      0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
      0.00000000e+00 0.00000000e+00 0.00000000e+00 1.00000000e+00
      0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
      0.00000000e+00 0.00000000e+00 0.00000000e+00]
     [3.20002745e-28 4.47876453e-16 2.05336047e-13 2.57698699e-15
      1.60570450e-18 1.19239951e-16 4.13435837e-27 5.76210344e-19
      2.74969042e-18 2.60153250e-22 3.14446299e-19 7.44190453e-28
      4.93604149e-18 4.13763885e-20 1.00000000e+00 6.17896183e-18
      2.94665265e-29 5.47696889e-34 3.13168431e-15 5.74484659e-36
      1.81712800e-27 8.39693919e-25 3.67593168e-24 9.58526377e-38
      2.19503463e-22 9.04472501e-17 4.70594730e-17 7.30607110e-28
      3.82860712e-18 5.37707707e-21 7.60952194e-29 9.65748175e-33
      4.13497361e-12 6.90422842e-20 6.29636288e-10 1.80472503e-15
      2.46801356e-19 1.90710538e-25 2.31073938e-20 3.34690079e-22
      4.06743898e-18 9.99225654e-24 1.67249852e-26]
     [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
      0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
      0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
      0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
      0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
      0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
      0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
      0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
      0.00000000e+00 1.00000000e+00 0.00000000e+00 0.00000000e+00
      1.53697036e-38 0.00000000e+00 0.00000000e+00 0.00000000e+00
      0.00000000e+00 0.00000000e+00 0.00000000e+00]]
    TopKV2(values=array([[1.00000000e+00, 7.07446657e-10, 4.28351937e-11, 2.27589193e-12,
            6.09216779e-13],
           [9.99205768e-01, 4.86577512e-04, 2.35112369e-04, 3.32113304e-05,
            1.42903655e-05],
           [1.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00],
           [1.00000000e+00, 6.29636288e-10, 4.13497361e-12, 2.05336047e-13,
            3.13168431e-15],
           [1.00000000e+00, 1.53697036e-38, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00]], dtype=float32), indices=array([[ 3,  2, 21, 10, 35],
           [31, 21,  3, 25, 11],
           [35,  0,  1,  2,  3],
           [14, 34, 32,  2, 18],
           [33, 36,  0,  1,  2]], dtype=int32))


### Project Writeup

Once you have completed the code implementation, document your results in a project writeup using this [template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) as a guide. The writeup can be in a markdown or pdf file. 

> **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  \n",
    "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.

---

## Step 4 (Optional): Visualize the Neural Network's State with Test Images

 This Section is not required to complete but acts as an additional excersise for understaning the output of a neural network's weights. While neural networks can be a great learning device they are often referred to as a black box. We can understand what the weights of a neural network look like better by plotting their feature maps. After successfully training your neural network you can see what it's feature maps look like by plotting the output of the network's weight layers in response to a test stimuli image. From these plotted feature maps, it's possible to see what characteristics of an image the network finds interesting. For a sign, maybe the inner network feature maps react with high activation to the sign's boundary outline or to the contrast in the sign's painted symbol.

 Provided for you below is the function code that allows you to get the visualization output of any tensorflow weight layer you want. The inputs to the function should be a stimuli image, one used during training or a new one you provided, and then the tensorflow variable name that represents the layer's state during the training process, for instance if you wanted to see what the [LeNet lab's](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) feature maps looked like for it's second convolutional layer you could enter conv2 as the tf_activation variable.

For an example of what feature map outputs look like, check out NVIDIA's results in their paper [End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) in the section Visualization of internal CNN State. NVIDIA was able to show that their network's inner weights had high activations to road boundary lines by comparing feature maps from an image with a clear path to one without. Try experimenting with a similar test to show that your trained network's weights are looking for interesting features, whether it's looking at differences in feature maps from images with or without a sign, or even what feature maps look like in a trained network vs a completely untrained one on the same sign image.

<figure>
 <img src="visualize_cnn.png" width="380" alt="Combined Image" />
 <figcaption>
 <p></p> 
 <p style="text-align: center;"> Your output should look something like this (above)</p> 
 </figcaption>
</figure>
 <p></p> 



```python
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
```
