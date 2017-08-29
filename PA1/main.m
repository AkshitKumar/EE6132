%% Initialization 
close all;
clear;
clc;

%% Load the training data
train_images = loadMNISTImages('train-images.idx3-ubyte');
train_labels = loadMNISTLabels('train-labels.idx1-ubyte');

%% Load the test data
test_images = loadMNISTImages('t10k-images.idx3-ubyte');
test_labels = loadMNISTLabels('t10k-labels.idx1-ubyte');

%% Neural Network Parameters
input_layer_size = size(train_images,1);
hidden_layer_1_size = 1000;
hidden_layer_2_size = 500;
hidden_layer_3_size = 250;
output_layer_size = 10;

%% Initialization of Weights 
weight_matrix_1 = randWeightInit(input_layer_size, hidden_layer_1_size);
weight_matrix_2 = randWeightInit(hidden_layer_1_size,hidden_layer_2_size);
weight_matrix_3 = randWeightInit(hidden_layer_2_size,hidden_layer_3_size);
weight_matrix_4 = randWeightInit(hidden_layer_3_size,output_layer_size);

%%% Storing the initial weight matrix for Relu operation
initial_weight_matrix_1 = weight_matrix_1;
initial_weight_matrix_2 = weight_matrix_2;
initial_weight_matrix_3 = weight_matrix_3;
initial_weight_matrix_4 = weight_matrix_4;

%%% Initialising the velocities
velocity_1 = zeroInitVelocity(input_layer_size, hidden_layer_1_size);
velocity_2 = zeroInitVelocity(hidden_layer_1_size, hidden_layer_2_size);
velocity_3 = zeroInitVelocity(hidden_layer_2_size, hidden_layer_3_size);
velocity_4 = zeroInitVelocity(hidden_layer_3_size,output_layer_size);

%% Training Parameters
lambda = 0.005;
step_size = 0.1;

%% Training the network
%%% Case 1 - Sigmoid with no alpha decay
activation = 0;
alpha_decay = 0;
[training_loss, test_loss, test_acc, learnt_weight_1, learnt_weight_2, learnt_weight_3, learnt_weight_4] = trainNN(8000,train_images,train_labels,test_images,test_labels,activation,lambda,step_size,weight_matrix_1,weight_matrix_2,weight_matrix_3,weight_matrix_4,alpha_decay,output_layer_size,velocity_1,velocity_2,velocity_3,velocity_4);
[estimate_value , estimates] = getTopThreeEstimates(learnt_weight_1, learnt_weight_2 , learnt_weight_3, learnt_weight_4, activation,test_images);
save 'case1_01.mat' training_loss test_loss test_acc estimate_value estimates


%%% Case 2 - Sigmoid with alpha decay
alpha_decay = 1;
[training_loss, test_loss, test_acc, learnt_weight_1, learnt_weight_2, learnt_weight_3, learnt_weight_4] = trainNN(8000,train_images,train_labels,test_images,test_labels,activation,lambda,step_size,weight_matrix_1,weight_matrix_2,weight_matrix_3,weight_matrix_4,alpha_decay,output_layer_size,velocity_1,velocity_2,velocity_3,velocity_4);
[estimate_value , estimates] = getTopThreeEstimates(learnt_weight_1, learnt_weight_2 , learnt_weight_3, learnt_weight_4, activation,test_images);
save 'case2_01.mat' training_loss test_loss test_acc estimate_value estimates

%%% Case 3 - ReLu with no alpha decay
activation = 1;
alpha_decay = 0;
[training_loss, test_loss, test_acc, learnt_weight_1, learnt_weight_2, learnt_weight_3, learnt_weight_4] = trainNN(8000,train_images,train_labels,test_images,test_labels,activation,lambda,step_size,weight_matrix_1,weight_matrix_2,weight_matrix_3,weight_matrix_4,alpha_decay,output_layer_size,velocity_1,velocity_2,velocity_3,velocity_4);
[estimate_value , estimates] = getTopThreeEstimates(learnt_weight_1, learnt_weight_2 , learnt_weight_3, learnt_weight_4, activation,test_images);
save 'case3_01.mat' training_loss test_loss test_acc estimate_value estimates

%%% Case 4 - ReLu with alpha decay
alpha_decay = 1;
[training_loss, test_loss, test_acc, learnt_weight_1, learnt_weight_2, learnt_weight_3, learnt_weight_4] = trainNN(8000,train_images,train_labels,test_images,test_labels,activation,lambda,step_size,weight_matrix_1,weight_matrix_2,weight_matrix_3,weight_matrix_4,alpha_decay,output_layer_size,velocity_1,velocity_2,velocity_3,velocity_4);
[estimate_value , estimates] = getTopThreeEstimates(learnt_weight_1, learnt_weight_2 , learnt_weight_3, learnt_weight_4, activation,test_images);
save 'case4_01.mat' training_loss test_loss test_acc estimate_value estimates