import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import re

data_files = glob.glob("mse_data/*.npz")

fig1 = plt.figure(1)
for file in data_files:
	data = np.load(file)
	LABEL = re.sub('mse_data/','',file)
	LABEL = re.sub('_', " ", LABEL)
	LABEL = re.sub('.npz',"",LABEL)
	plt.plot(np.arange(data['tl'].shape[0]), data['tl'], label = LABEL)
plt.title('Training Loss vs Number of Iterations')
plt.xlabel('Number of Iterations')
plt.ylabel('Training Loss')
plt.legend()

fig2 = plt.figure(2)
for file in data_files:
	data = np.load(file)
	LABEL = re.sub('mse_data/','',file)
	LABEL = re.sub('_', " ", LABEL)
	LABEL = re.sub('.npz',"",LABEL)
	plt.plot(np.arange(data['ta'].shape[0]), data['ta'], label = LABEL)
plt.title('Training Accuracy vs Number of Iterations')
plt.xlabel('Number of Iterations')
plt.ylabel('Training Accuracy')
plt.legend()

fig3 = plt.figure(3)
for file in data_files:
	data = np.load(file)
	LABEL = re.sub('mse_data/','',file)
	LABEL = re.sub('_', " ", LABEL)
	LABEL = re.sub('.npz',"",LABEL)
	plt.plot(np.arange(data['tsl'].shape[0]), data['tsl'], label = LABEL)
plt.title('Test Loss vs Number of Iterations')
plt.xlabel('Number of Iterations')
plt.ylabel('Test Loss')
plt.legend()

fig4 = plt.figure(4)
for file in data_files:
	data = np.load(file)
	LABEL = re.sub('mse_data/','',file)
	LABEL = re.sub('_', " ", LABEL)
	LABEL = re.sub('.npz',"",LABEL)
	plt.plot(np.arange(data['tsa'].shape[0]), data['tsa'], label = LABEL)
plt.title('Test Accuracy vs Number of Iterations')
plt.xlabel('Number of Iterations')
plt.ylabel('Test Accuracy')
plt.legend()

bit_accuracies = np.zeros(20)

fig5 = plt.figure(5)
for file in data_files:
	data = np.load(file)
	LABEL = re.sub('mse_data/','',file)
	LABEL = re.sub('_digits.npz', " ", LABEL)
	index = int(LABEL)
	bit_accuracies[index - 1] = data['tsa'][-1]
plt.plot([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],bit_accuracies,'ro')
plt.xlabel(r'Bit Length ($L$)')
plt.ylabel('Average Bit Accuracies')
plt.title(r'Avg Bit Accuracies vs Input Length($L$)')

plt.show()