import numpy as np
import matplotlib.pyplot as plt

vanilla_rnn = np.load('vanilla_rnn.npz')
lstm_rnn = np.load('lstm_rnn.npz')
bi_rnn = np.load('bi_rnn.npz')

fig1 = plt.figure(1)
plt.plot(np.arange(vanilla_rnn['tl'].shape[0]),vanilla_rnn['tl'],np.arange(lstm_rnn['tl'].shape[0]),lstm_rnn['tl'], np.arange(bi_rnn
	['tl'].shape[0]),bi_rnn['tl'])
plt.xlabel('Number of Iterations')
plt.ylabel('Training Loss')
plt.title('Training Loss vs Number of Iterations')
plt.legend(["Vanilla RNN", "LSTM RNN", "Bi-directional LSTM"])
fig1.savefig('plots/training_loss.png')

fig2 = plt.figure(2)
plt.plot(np.arange(vanilla_rnn['ta'].shape[0]),vanilla_rnn['ta'],np.arange(lstm_rnn['ta'].shape[0]),lstm_rnn['ta'], np.arange(bi_rnn
	['ta'].shape[0]),bi_rnn['ta'])
plt.xlabel('Number of Iterations')
plt.ylabel('Training Accuracy')
plt.title('Training Accuracy vs Number of Iterations')
plt.legend(["Vanilla RNN", "LSTM RNN", "Bi-directional LSTM"])
fig2.savefig('plots/training_accuracy.png')

fig3 = plt.figure(3)
plt.plot(np.arange(vanilla_rnn['vl'].shape[0]),vanilla_rnn['vl'],np.arange(lstm_rnn['vl'].shape[0]),lstm_rnn['vl'], np.arange(bi_rnn
	['vl'].shape[0]),bi_rnn['vl'])
plt.xlabel('Number of Iterations')
plt.ylabel('Validation Loss')
plt.title('Validation Loss vs Number of Iterations')
plt.legend(["Vanilla RNN", "LSTM RNN", "Bi-directional LSTM"])
fig3.savefig('plots/validation_loss.png')

fig4 = plt.figure(4)
plt.plot(np.arange(vanilla_rnn['va'].shape[0]),vanilla_rnn['va'],np.arange(lstm_rnn['va'].shape[0]),lstm_rnn['va'], np.arange(bi_rnn
	['va'].shape[0]),bi_rnn['va'])
plt.xlabel('Number of Iterations')
plt.ylabel('Validation Accuracy')
plt.title('Validation Accuracy vs Number of Iterations')
plt.legend(["Vanilla RNN", "LSTM RNN", "Bi-directional LSTM"])
fig1.savefig('plots/validation_accuracy.png')

plt.show()
