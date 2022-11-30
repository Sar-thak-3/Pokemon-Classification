import os
from pathlib import Path
from keras.utils import load_img, img_to_array
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
p = Path("./set/")
dirs = p.glob("*")

image_data = []
labels = []
labels_dict = {"pikachu": 0,"bulbasaur": 1,"charmander": 2,"squirtle": 3}

for folder_dir in dirs:
    folder_name = str(folder_dir).split("\\")[-1]
    print(folder_name)
    count = 0

    # Iterate over folder directory and pick all images of the pokemons
    for img_path in folder_dir.glob("*.jpg"):
        img = load_img(img_path,target_size=(40,40))
        img_array = img_to_array(img)
        image_data.append(img_array)
        labels.append(labels_dict[folder_name])
        count += 1
    print(count)

labels = np.array(labels)
image_data = np.array(image_data)/255.0

def drawImg(img):
    plt.imshow(img)
    plt.show()

from sklearn.utils import shuffle
x,y = shuffle(image_data,labels,random_state=2)
# print(x.shape,y.shape)
drawImg(x[0])

# Split training and testing data
split = int(x.shape[0]*0.8)
m = x.shape[0]
x_train = x[:split,:]
x_train = x_train.reshape(split,-1)
y_train = y[:split]
x_test = x[split:,:]
x_test = x_test.reshape(m-split,-1)
y_test = y[split:]

print(x_train.shape,x_test.shape)

def softmax(a):
    e_pa = np.exp(a)
    ans = e_pa/np.sum(e_pa,axis=1,keepdims=True) # keepdims=True -> preserves the shape of the np.array
    return ans

# Neural Network
class NeuralNetwork:
    def __init__(self,input_size,layers,output_size):
        np.random.seed(0)
        model = {}
        
        model['w1'] = np.random.randn(input_size,layers[0])
        model['b1'] = np.zeros((1,layers[0]))

        model['w2'] = np.random.randn(layers[0],layers[1])
        model['b2'] = np.zeros((1,layers[1]))

        model['w3'] = np.random.randn(layers[1],output_size)
        model['b3'] = np.zeros((1,output_size))

        self.model = model

    def forward(self,x):
        w1,w2,w3 = self.model['w1'],self.model['w2'],self.model['w3']
        b1,b2,b3 = self.model['b1'],self.model['b2'],self.model['b3']

        z1 = np.dot(x,w1) + b1
        a1 = np.tanh(z1)

        z2 = np.dot(a1,w2) + b2
        a2 = np.tanh(z2)

        z3 = np.dot(a2,w3) + b3
        y_ = softmax(z3)

        self.activation_outputs = (a1,a2,y_)
        return y_
    
    def backward(self,x,y,learning_rate=0.001):
        w1,w2,w3 = self.model['w1'],self.model['w2'],self.model['w3']
        b1,b2,b3 = self.model['b1'],self.model['b2'],self.model['b3']

        a1,a2,y_ = self.activation_outputs

        delta3 = y_-y
        dw3 = np.dot(a2.T,delta3)
        db3 = np.sum(delta3,axis=0)

        delta2 = (1-np.square(a2))*np.dot(delta3,w3.T)
        dw2 = np.dot(a1.T,delta2)
        db2 = np.sum(delta2,axis=0)

        delta1 = (1-np.square(a1))*np.dot(delta2,w2.T)
        dw1 = np.dot(x.T,delta1)
        db1 = np.sum(delta1,axis=0)

        self.model['w1'] -= learning_rate*dw1
        self.model['b1'] -= learning_rate*db1

        self.model['w2'] -= learning_rate*dw2
        self.model['b2'] -= learning_rate*db2

        self.model['w3'] -= learning_rate*dw3
        self.model['b3'] -= learning_rate*db3

    def predict(self,x):
        y_out = self.forward(x)
        return np.argmax(y_out,axis=1)

    def summary(self):
        w1,w2,w3 = self.model['w1'],self.model['w2'] , self.model['w3']
        b1,b2,b3 = self.model['b1'],self.model['b2'],self.model['b3']

        a1,a2,y_ = self.activation_outputs

        print("W1" ,w1.shape)
        print("a1",a1.shape)

        print("W2" ,w2.shape)
        print("a2",a2.shape)

        print("W3" ,w3.shape)
        print("y",y_.shape)

def loss(y_oht,p):
    l = -np.mean(y_oht*np.log(p))
    return l

def oneHot(y,depth):
    m = y.shape[0]
    y_oht = np.zeros((m,depth))
    y_oht[np.arange(m),y] = 1
    return y_oht

def train(x,y,model,epochs,learning_rate,logs=True):
    training_loss = []
    classes = 4
    y_OHT = oneHot(y,classes)

    for ix in range(epochs):
        y_ = model.forward(x)
        l = loss(y_OHT,y_)
        training_loss.append(l)
        model.backward(x,y_OHT,learning_rate)
        if(logs):
            print("Epoch",ix,"Loss",l)

    return training_loss

model = NeuralNetwork(input_size=4800,layers=[100,50],output_size=4)
l = train(x_train,y_train,model,100,0.0002)

from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
outputs = model.predict(x_test)
training_accuracy = np.sum(outputs==y_test)/y_test.shape[0]
print(training_accuracy*100)
# 64.47368421052632
cnf_matrix = confusion_matrix(outputs,y_test)
print(cnf_matrix)

plot_confusion_matrix(cnf_matrix,class_names=['pikachu','bulbasaur','charmander','squirtle'])
plt.show()
# this plot tells the accuracy and classification report in form of graph!