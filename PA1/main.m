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
weight_matrix_4 = randWeightInit(hidden_layer_3_size,output_layer);

%% Training the neural network
for i = 1:10000
    r = randi([1 size(train_images,2)],1,64);
    X = train_images(:,r);
    y = train_labels(:,r);
    [weight_grad_1 , weight_grad_2 , weight_grad_3 , weight_grad_4] = calculateGradient(X,y,weight_matrix_1,weight_matrix_2, weight_matrix_3, weight_matrix_4);
    
    
    
    