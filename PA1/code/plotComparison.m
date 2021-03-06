%% Loading the data
case1 = load('case1_01.mat');
case2 = load('case2_01.mat');
case3 = load('case3_01.mat');
case4 = load('case4_01.mat');

%% Plotting the training loss function
%{
%%% Constant Learning Rate comparison - Training Loss
figure();
t = 1 : length(case2.training_loss);
plot(t,case2.training_loss,t,case4.training_loss);
title('Comparison of Sigmoid and ReLu for Scheduled Learning Rate');
xlabel('Iterations');
ylabel('Training Loss');
legend('Sigmoid','ReLu');

%%% Constant Learning Rate comparison - test loss
figure();
t = 1 : length(case2.test_loss);
plot(t,case2.test_loss,t,case4.test_loss);
title('Comparison of Sigmoid and ReLu for Scheduled Learning Rate');
xlabel('Iterations');
ylabel('Test Loss');
legend('Sigmoid','ReLu');

%%% Constant Learning Rate Comparison - test acc
figure();
t = 1 : length(case2.test_acc);
plot(t,case2.test_acc,t,case4.test_acc);
title('Comparison of Sigmoid and ReLu for Scheduled Learning Rate');
xlabel('Iterations');
ylabel('Test Accuracy');
legend('Sigmoid','ReLu');


%% Plotting the comparison between schedule and constant
figure();
subplot(2,1,1);
t = 1 : length(case1.training_loss);
plot(t,case1.training_loss,t,case2.training_loss);
title('Comparison of Scheduled Rate vs Constant Rate for Sigmoid Function')
xlabel('Iterations');
ylabel('Training Loss');
legend('Constant Rate','Scheduled Rate');

subplot(2,1,2);
plot(t,case3.training_loss,t,case4.training_loss);
title('Comparison of Scheduled Rate vs Constant Rate for ReLu Function')
xlabel('Iterations');
ylabel('Training Loss');
legend('Constant Rate','Scheduled Rate');

figure();
subplot(2,1,1);
t = 1 : length(case1.test_loss);
plot(t,case1.test_loss,t,case2.test_loss);
title('Comparison of Scheduled Rate vs Constant Rate for Sigmoid Function')
xlabel('Iterations');
ylabel('Test Loss');
legend('Constant Rate','Scheduled Rate');

subplot(2,1,2);
plot(t,case3.test_loss,t,case4.test_loss);
title('Comparison of Scheduled Rate vs Constant Rate for ReLu Function')
xlabel('Iterations');
ylabel('Test Loss');
legend('Constant Rate','Scheduled Rate');

figure();
subplot(2,1,1);
t = 1 : length(case1.test_acc);
plot(t,case1.test_acc,t,case2.test_acc);
title('Comparison of Scheduled Rate vs Constant Rate for Sigmoid Function')
xlabel('Iterations');
ylabel('Test Accuracy');
legend('Constant Rate','Scheduled Rate');

subplot(2,1,2);
plot(t,case3.test_acc,t,case4.test_acc);
title('Comparison of Scheduled Rate vs Constant Rate for ReLu Function')
xlabel('Iterations');
ylabel('Test Accuracy');
legend('Constant Rate','Scheduled Rate');
%}

%% Plotting of comparison of learning rates 
case_01_sigmoid = load('case_01_sigmoid.mat');
case_01_relu = load('case_01_relu.mat');

case_001_sigmoid = load('case_001_sigmoid.mat');
case_001_relu = load('case_001_relu.mat');

case_0001_sigmoid = load('case_0001_sigmoid.mat');
case_0001_relu = load('case_0001_relu.mat');

figure();
subplot(2,1,1);
t = 1:length(case_01_sigmoid.training_loss);
plot(t,case_01_sigmoid.training_loss,t,case_001_sigmoid.training_loss,t,case_0001_sigmoid.training_loss);
title('Comparison of Different Learning Rates with Sigmoid Activation');
xlabel('Iterations');
ylabel('Training Loss');
legend('lr = 0.1','lr=0.01','lr=0.001');

subplot(2,1,2);
plot(t,case_01_relu.training_loss,t,case_001_relu.training_loss,t,case_0001_relu.training_loss);
title('Comparison of Different Learning Rates with ReLu Activation');
xlabel('Iterations');
ylabel('Training Loss');
legend('lr = 0.1','lr=0.01','lr=0.001');

figure();
subplot(2,1,1);
t = 1:length(case_01_sigmoid.test_loss);
plot(t,case_01_sigmoid.test_loss,t,case_001_sigmoid.test_loss,t,case_0001_sigmoid.test_loss);
title('Comparison of Different Learning Rates with Sigmoid Activation');
xlabel('Iterations');
ylabel('Test Loss');
legend('lr = 0.1','lr=0.01','lr=0.001');

subplot(2,1,2);
plot(t,case_01_relu.test_loss,t,case_001_relu.test_loss,t,case_0001_relu.test_loss);
title('Comparison of Different Learning Rates with ReLu Activation');
xlabel('Iterations');
ylabel('Test Loss');
legend('lr = 0.1','lr=0.01','lr=0.001');



figure();
subplot(2,1,1);
t = 1:length(case_01_sigmoid.test_acc);
plot(t,case_01_sigmoid.test_acc,t,case_001_sigmoid.test_acc,t,case_0001_sigmoid.test_acc);
title('Comparison of Different Learning Rates with Sigmoid Activation');
xlabel('Iterations');
ylabel('Test Accuracy');
legend('lr = 0.1','lr=0.01','lr=0.001');

subplot(2,1,2);
plot(t,case_01_relu.test_acc,t,case_001_relu.test_acc,t,case_0001_relu.test_acc);
title('Comparison of Different Learning Rates with ReLu Activation');
xlabel('Iterations');
ylabel('Test Accuracy');
legend('lr = 0.1','lr=0.01','lr=0.001');

