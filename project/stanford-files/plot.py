import numpy as np
import matplotlib.pyplot as plt

vanilla_data = np.load('dropout_data.npz')
vanilla_inc_data = np.load('aggresive_dropout_data.npz')

t = vanilla_data['arr_0'].shape[0]

plt.plot(range(1,t+1), vanilla_data['arr_0'], range(1,t+1), vanilla_data['arr_2'], range(1,t+1) , vanilla_inc_data['arr_0'], range(1,t+1), vanilla_inc_data['arr_2'])
plt.xlabel('No. of Epochs')
plt.xticks([5,10,15])
plt.ylabel('Loss')
plt.legend(["Training Loss (Dropout - 0.2)", "Validation Loss (Dropout - 0.2)", "Training Loss (Dropout - 0.5)", "Validation Loss (Dropout - 0.5)"])

plt.figure()

plt.plot(range(1,t+1), vanilla_data['arr_1'], range(1,t+1), vanilla_data['arr_3'], range(1,t+1) , vanilla_inc_data['arr_1'], range(1,t+1), vanilla_inc_data['arr_3'])
plt.xlabel('No. of Epochs')
plt.xticks([5,10,15])
plt.ylabel('Accuracy')
plt.legend(["Training Accuracy (Dropout - 0.2)", "Validation Accuracy (Dropout - 0.2)", "Training Accuracy (Dropout - 0.5)", "Validation Accuracy (Dropout - 0.5)"])


plt.show()