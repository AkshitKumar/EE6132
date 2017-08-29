%% Loading the data
sigmoid_tl_no_decay = load('sigmoid_s1_tl_no_decay.mat');
sigmoid_testl_no_decay = load('sigmoid_s1_testl_no_decay.mat');
sigmoid_testacc_no_decay = load('sigmoid_s1_testacc_no_decay.mat');

sigmoid_tl_decay = load('sigmoid_s1_tl_decay.mat');
sigmoid_testl_decay = load('sigmoid_s1_testl_decay.mat');
sigmoid_testacc_decay = load('sigmoid_s1_testacc_decay.mat');

x = length(sigmoid_tl_no_decay.training_loss);
plot(1:x,sigmoid_tl_no_decay.training_loss,1:x,sigmoid_tl_decay.training_loss);
legend('No Decay', 'Decay');