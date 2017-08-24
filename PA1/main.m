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

%% Training Parameters
lambda = 0.005;
step_size = 0.1;
% lossfunction = [];
% test_loss_trend = [];
% test_acc = [];
% 
% lossfunctionrelu = [];
% test_loss_relu = [];

[training_loss, test_loss, test_acc] = trainNN(8000,train_images,train_labels,test_images,test_labels,1,lambda,step_size,weight_matrix_1,weight_matrix_2,weight_matrix_3,weight_matrix_4,0,output_layer_size);

%{
%% Training the neural network using sigmoid activation function
for i = 1:10000
    r = randi([1 size(train_images,2)],1,64);
    X = train_images(:,r);
    y = train_labels(r,:);
    [loss,weight_grad_1 , weight_grad_2 , weight_grad_3 , weight_grad_4] = calculateGradient(X',y,output_layer_size,lambda,weight_matrix_1,weight_matrix_2, weight_matrix_3, weight_matrix_4); 
    [weight_matrix_1, weight_matrix_2 , weight_matrix_3, weight_matrix_4] = updateWeightMatrix(weight_matrix_1,weight_grad_1,weight_matrix_2,weight_grad_2,weight_matrix_3,weight_grad_3,weight_matrix_4,weight_grad_4,step_size);
    lossfunction = [lossfunction; loss];
    if mod(i,200) == 0
        [test_loss,acc] = calculateTestLoss(test_images',test_labels,output_layer_size,lambda,weight_matrix_1,weight_matrix_2,weight_matrix_3,weight_matrix_4);
        test_loss_trend = [test_loss_trend ; test_loss];
        test_acc = [test_acc; acc];
    end
end


%% Reinitializing the weight matrices
weight_matrix_1 = initial_weight_matrix_1;
weight_matrix_2 = initial_weight_matrix_2;
weight_matrix_3 = initial_weight_matrix_3;
weight_matrix_4 = initial_weight_matrix_4;


%% Training the Neural Network using ReLu activation function
for i = 1:10000
    r = randi([1 size(train_images,2)],1,64);
    X = train_images(:,r);
    y = train_labels(r,:);
    [loss,weight_grad_1 , weight_grad_2 , weight_grad_3 , weight_grad_4] = calculateGradientRelu(X',y,output_layer_size,lambda,weight_matrix_1,weight_matrix_2, weight_matrix_3, weight_matrix_4); 
    [weight_matrix_1, weight_matrix_2 , weight_matrix_3, weight_matrix_4] = updateWeightMatrix(weight_matrix_1,weight_grad_1,weight_matrix_2,weight_grad_2,weight_matrix_3,weight_grad_3,weight_matrix_4,weight_grad_4,step_size);
    lossfunctionrelu = [lossfunctionrelu; loss];
    if mod(i,200) == 0
        test_loss = calculateTestLossRelu(test_images',test_labels,output_layer_size,lambda,weight_matrix_1,weight_matrix_2,weight_matrix_3,weight_matrix_4);
        test_loss_relu = [test_loss_relu ; test_loss];
    end
end
%}


