# Classifying Text with Neural Networks and TensorFlow
Hi! I was having a hard time learning how to use TensorFlow, so I got angry and I decided that I would learn how to use it and that I would also share the knowledge to help other people as well :)

You can check the blog post [here](https://medium.freecodecamp.org/big-picture-machine-learning-classifying-text-with-neural-networks-and-tensorflow-d94036ac2274) (there are more details about Neural Networks there) or if you are in a hurry check just check the code below ;)

***

```python
import pandas as pd
import numpy as np
import tensorflow as tf
from collections import Counter
from sklearn.datasets import fetch_20newsgroups
```

## How TensorFlow works


```python
import tensorflow as tf
my_graph = tf.Graph()
with tf.Session(graph=my_graph) as sess:
    x = tf.constant([1,3,6]) 
    y = tf.constant([1,1,1])
    op = tf.add(x,y)
    result = sess.run(fetches=op)
    print(result)
```

    [2 4 7]


## How to manipulate data and pass it to the Neural Network inputs


```python
vocab = Counter()

text = "Hi from Brazil"

for word in text.split(' '):
    word_lowercase = word.lower()
    vocab[word_lowercase]+=1
        
def get_word_2_index(vocab):
    word2index = {}
    for i,word in enumerate(vocab):
        word2index[word] = i
        
    return word2index
```


```python
word2index = get_word_2_index(vocab)

total_words = len(vocab)
matrix = np.zeros((total_words),dtype=float)

for word in text.split():
    matrix[word2index[word.lower()]] += 1
    
print("Hi from Brazil:", matrix)
```

    Hi from Brazil: [ 1.  1.  1.]



```python
matrix = np.zeros((total_words),dtype=float)
text = "Hi"
for word in text.split():
    matrix[word2index[word.lower()]] += 1
    
print("Hi:", matrix)
```

    Hi: [ 1.  0.  0.]


## Building the neural network


```python
categories = ["comp.graphics","sci.space","rec.sport.baseball"]
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)
```


```python
print('total texts in train:',len(newsgroups_train.data))
print('total texts in test:',len(newsgroups_test.data))
```

    total texts in train: 1774
    total texts in test: 1180



```python
print('text',newsgroups_train.data[0])
print('category:',newsgroups_train.target[0])
```

    text From: jk87377@lehtori.cc.tut.fi (Kouhia Juhana)
    Subject: Re: More gray levels out of the screen
    Organization: Tampere University of Technology
    Lines: 21
    Distribution: inet
    NNTP-Posting-Host: cc.tut.fi
    
    In article <1993Apr6.011605.909@cis.uab.edu> sloan@cis.uab.edu
    (Kenneth Sloan) writes:
    >
    >Why didn't you create 8 grey-level images, and display them for
    >1,2,4,8,16,32,64,128... time slices?
    
    By '8 grey level images' you mean 8 items of 1bit images?
    It does work(!), but it doesn't work if you have more than 1bit
    in your screen and if the screen intensity is non-linear.
    
    With 2 bit per pixel; there could be 1*c_1 + 4*c_2 timing,
    this gives 16 levels, but they are linear if screen intensity is
    linear.
    With 1*c_1 + 2*c_2 it works, but we have to find the best
    compinations -- there's 10 levels, but 16 choises; best 10 must be
    chosen. Different compinations for the same level, varies a bit, but
    the levels keeps their order.
    
    Readers should verify what I wrote... :-)
    
    Juhana Kouhia
    
    category: 0



```python
vocab = Counter()

for text in newsgroups_train.data:
    for word in text.split(' '):
        vocab[word.lower()]+=1
        
for text in newsgroups_test.data:
    for word in text.split(' '):
        vocab[word.lower()]+=1
```


```python
print("Total words:",len(vocab))
```

    Total words: 119930



```python
total_words = len(vocab)

def get_word_2_index(vocab):
    word2index = {}
    for i,word in enumerate(vocab):
        word2index[word.lower()] = i
        
    return word2index

word2index = get_word_2_index(vocab)

print("Index of the word 'the':",word2index['the'])
```

    Index of the word 'the': 10



```python
def text_to_vector(text):
    layer = np.zeros(total_words,dtype=float)
    for word in text.split(' '):
        layer[word2index[word.lower()]] += 1
        
    return layer

def category_to_vector(category):
    y = np.zeros((3),dtype=float)
    if category == 0:
        y[0] = 1.
    elif category == 1:
        y[1] = 1.
    else:
        y[2] = 1.
        
    return y
```


```python
def get_batch(df,i,batch_size):
    batches = []
    results = []
    texts = df.data[i*batch_size:i*batch_size+batch_size]
    categories = df.target[i*batch_size:i*batch_size+batch_size]
    
    for text in texts:
        layer = text_to_vector(text) 
        batches.append(layer)
        
    for category in categories:
        y = category_to_vector(category)
        results.append(y)  
     
    return np.array(batches),np.array(results)
```


```python
print("Each batch has 100 texts and each matrix has 119930 elements (words):",get_batch(newsgroups_train,1,100)[0].shape)
```

    Each batch has 100 texts and each matrix has 119930 elements (words): (100, 119930)



```python
print("Each batch has 100 labels and each matrix has 3 elements (3 categories):",get_batch(newsgroups_train,1,100)[1].shape)
```

    Each batch has 100 labels and each matrix has 3 elements (3 categories): (100, 3)



```python
# Parameters
learning_rate = 0.01
training_epochs = 10
batch_size = 150
display_step = 1

# Network Parameters
n_hidden_1 = 100      # 1st layer number of features
n_hidden_2 = 100       # 2nd layer number of features
n_input = total_words # Words in vocab
n_classes = 3         # Categories: graphics, sci.space and baseball

input_tensor = tf.placeholder(tf.float32,[None, n_input],name="input")
output_tensor = tf.placeholder(tf.float32,[None, n_classes],name="output") 
```


```python
def multilayer_perceptron(input_tensor, weights, biases):
    layer_1_multiplication = tf.matmul(input_tensor, weights['h1'])
    layer_1_addition = tf.add(layer_1_multiplication, biases['b1'])
    layer_1 = tf.nn.relu(layer_1_addition)
    
    # Hidden layer with RELU activation
    layer_2_multiplication = tf.matmul(layer_1, weights['h2'])
    layer_2_addition = tf.add(layer_2_multiplication, biases['b2'])
    layer_2 = tf.nn.relu(layer_2_addition)
    
    # Output layer 
    out_layer_multiplication = tf.matmul(layer_2, weights['out'])
    out_layer_addition = out_layer_multiplication + biases['out']
    
    return out_layer_addition
```


```python
# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
prediction = multilayer_perceptron(input_tensor, weights, biases)

# Define loss and optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=output_tensor))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Initializing the variables
init = tf.global_variables_initializer()

# [NEW] Add ops to save and restore all the variables
saver = tf.train.Saver()
```


```python
# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(newsgroups_train.data)/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x,batch_y = get_batch(newsgroups_train,i,batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            c,_ = sess.run([loss,optimizer], feed_dict={input_tensor: batch_x,output_tensor:batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "loss=", \
                "{:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(output_tensor, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    total_test_data = len(newsgroups_test.target)
    batch_x_test,batch_y_test = get_batch(newsgroups_test,0,total_test_data)
    print("Accuracy:", accuracy.eval({input_tensor: batch_x_test, output_tensor: batch_y_test}))
    
    # [NEW] Save the variables to disk
    save_path = saver.save(sess, "/tmp/model.ckpt")
    print("Model saved in path: %s" % save_path)
```

    Epoch: 0001 loss= 1289.632795854
    Epoch: 0002 loss= 270.930032210
    Epoch: 0003 loss= 220.176122665
    Epoch: 0004 loss= 119.421511043
    Epoch: 0005 loss= 47.321779143
    Epoch: 0006 loss= 32.028330424
    Epoch: 0007 loss= 11.666836267
    Epoch: 0008 loss= 15.337957333
    Epoch: 0009 loss= 6.931441394
    Epoch: 0010 loss= 0.000000000
    Optimization Finished!
    Accuracy: 0.749153
    Model saved in path: /tmp/model.ckpt


## Making a prediction

### 1 text


```python
# Get a text to make a prediction
text_for_prediction = newsgroups_test.data[5]
```


```python
print('text',text_for_prediction)
```

    text From: neff@iaiowa.physics.uiowa.edu (John S. Neff)
    Subject: Re: Space spinn offs
    Nntp-Posting-Host: pluto.physics.uiowa.edu
    Organization: The University of Iowa
    Lines: 23
    
    In article <1rruis$9do@bigboote.WPI.EDU> wfbrown@wpi.WPI.EDU (William F Brown) writes:
    >From: wfbrown@wpi.WPI.EDU (William F Brown)
    >Subject: Re: Space spinn offs
    >Date: 30 Apr 1993 19:27:24 GMT
    >I just wanted to point out, that Teflon wasn't from the space program.
    >It was from the WWII nuclear weapons development program.  Pipes in the 
    >system for fractioning and enriching uranium had to be lined with it.
    >
    >Uranium Hexafloride was the chemical they turned the pitchblend into for
    >enrichment.  It is massively corrosive.  Even to Stainless steels. Hence
    >the need for a very inert substaance to line the pipes with.  Teflon has
    >all its molecular sockets bound up already, so it is very unreactive.
    >
    >My 2 sense worth.
    >
    >Bill
    >
    
    The artifical pacemaker was invented in 1958 by Wilson Greatbatch an
    American biomedical engineer. The bill authorizing NASA was signed
    in October of 1958 so it is clear that NASA had nothing to do with
    the invention of the pacemaker.
    
    



```python
print("Text correct category:", newsgroups_test.target[5])
```

    Text correct category: 2



```python
# Convert text to vector so we can send it to our model
vector_txt = text_to_vector(text)
# Wrap vector like we do in get_batches()
input_array = np.array([vector_txt])
```


```python
input_array.shape
```




    (1, 119930)




```python
saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, "/tmp/model.ckpt")
    print("Model restored.")
    
    classification = sess.run(tf.argmax(prediction, 1), feed_dict={input_tensor: input_array})
    print("Predicted category:", classification)
```

    Model restored.
    Predicted category: [0]


### 10 texts


```python
# Get 10 texts to make a prediction

x_10_texts,y_10_correct_labels = get_batch(newsgroups_test,0,10)

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, "/tmp/model.ckpt")
    print("Model restored.")
    
    classification = sess.run(tf.argmax(prediction, 1), feed_dict={input_tensor: x_10_texts})
    print("Predicted categories:", classification)
```

    Model restored.
    Predicted categories: [1 0 2 0 1 2 1 0 1 2]



```python
print("Correct categories:", np.argmax(y_10_correct_labels, 1))
```

    Correct categories: [1 0 1 0 1 2 2 0 1 2]

