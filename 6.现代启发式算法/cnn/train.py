import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets
from conv3x3 import Conv3x3
from maxpool2 import MaxPool2
from softmax import Softmax
 
# 加载MNIST数据集
mnist = datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

conv = Conv3x3(8)                                    # 28x28x1 -> 26x26x8
pool = MaxPool2()                                    # 26x26x8 -> 13x13x8
softmax = Softmax(13 * 13 * 8, 10)                   # 13x13x8 -> 10
print('MNIST CNN initialized!')
 
def forward(image, label):
    '''
    Completes a forward pass of the CNN and calculates the accuracy and
    cross-entropy loss.\n
    Parameters:
    ---
    - image is a 2d numpy array\n
    - label is a digit\n
    Returns:
    ---
    - out: NumPy数组，包含预测出每个类型的概率
    - loss: float，交叉熵损失
    - acc: 0或1，表示当前预测的结果，0表示错误，1表示正确
    '''
    ####################
    # 网络结构
    ####################
    # We transform the image from [0, 255] to [-0.5, 0.5] to make it easier
    # to work with. This is standard practice.
    out = conv.forward((image / 255) - 0.5)
    out = pool.forward(out)
    out = softmax.forward(out)
 
    ####################
    # 损失函数和准确率
    ####################
    # Calculate cross-entropy loss and accuracy. np.log() is the natural log.
    loss = -np.log(out[label])
    acc = 1 if np.argmax(out) == label else 0
 
    return out, loss, acc
    # out: vertor of probability
    # loss: num
    # acc: 1 or 0
 
def train(im, label, lr=.005):
    '''
    Completes a full training step on the given image and label.
    Returns the cross-entropy loss and accuracy.
    - image is a 2d numpy array
    - label is a digit
    - lr is the learning rate
    '''
    # Forward
    out, loss, acc = forward(im, label)
 
    # Calculate initial gradient
    gradient = np.zeros(10)
    gradient[label] = -1 / out[label]
 
    # Backprop
    gradient = softmax.backprop(gradient, lr)
    gradient = pool.backprop(gradient)
    gradient = conv.backprop(gradient, lr)
 
    return loss, acc

 
# Train the CNN for 3 epochs
for epoch in range(3):
    print('--- Epoch %d ---' % (epoch + 1))
 
    # Shuffle the training data
    permutation = np.random.permutation(len(train_images))# 随机生成读取训练数据的顺序
    train_images = train_images[permutation]# 按permutation中的顺序打乱
    train_labels = train_labels[permutation]
 
    # Train!
    loss = 0
    num_correct = 0
 
    # i: index
    # im: image
    # label: label
    for i, (im, label) in enumerate(zip(train_images, train_labels)):
        if i > 0 and i % 100 == 99:
            print(
                '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
                (i + 1, loss / 100, num_correct)
            )
            loss = 0
            num_correct = 0
 
        l, acc = train(im, label)
        loss += l
        num_correct += acc
 
# Test the CNN
print('\n--- Testing the CNN ---')
loss = 0
num_correct = 0
for im, label in zip(test_images, test_labels):
    _, l, acc = forward(im, label)
    loss += l
    num_correct += acc
 
num_tests = len(test_images)
print('Test Loss:', loss / num_tests)
print('Test Accuracy:', num_correct / num_tests)