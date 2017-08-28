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
step_size = 0.01;

%% Training the network
activation = 0;
alpha_decay = 0;
[training_loss, test_loss, test_acc] = trainNN(8000,train_images,train_labels,test_images,test_labels,activation,lambda,step_size,weight_matrix_1,weight_matrix_2,weight_matrix_3,weight_matrix_4,alpha_decay,output_layer_size,velocity_1,velocity_2,velocity_3,velocity_4);
