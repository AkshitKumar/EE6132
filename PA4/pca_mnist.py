from sklearn.decomposition import PCA
from sklearn.datasets import fetch_mldata
import numpy as np
import matplotlib.pyplot as plt
from random import *

mnist = fetch_mldata("MNIST Original")
x_train, y_train = mnist.data[:70000]/ 255.0 , mnist.target[:70000]

pca = PCA(n_components = 30)
pca.fit(x_train)

x_train_pca = pca.transform(x_train)
reconstructed_x_train_pca = pca.inverse_transform(x_train_pca)

for i in range(10):
    x = randint(1,70000)
    ax = plt.subplot(2, 10, i+1)
    plt.imshow(x_train[x].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, 10, i+1+10)
    plt.imshow(reconstructed_x_train_pca[x].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

print ((x_train - reconstructed_x_train_pca) ** 2).mean()
plt.show()
