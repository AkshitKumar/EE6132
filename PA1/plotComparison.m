%% Loading the data
case1 = load('case1_03.mat');
case2 = load('case2_03.mat');
case3 = load('case3_03.mat');
case4 = load('case4_03.mat');

%% Plotting the training loss function
%%% Constant Learning Rate comparison
figure();
t = 1 : length(case1.training_loss);
plot(t,case1.training_loss,t,case3.training_loss);
title('Comparison of Sigmoid and ReLu for Constant Learning Rate');
xlabel('Iterations');
ylabel('Training Loss');
legend('Sigmoid','ReLu');
