function [grad_1,grad_2,grad_3, grad_4] = calculateGradient(X,y,weight_matrix_1,weight_matrix_2,weight_matrix_3,weight_matrix_4)
    %% Forward Propogation
    a1 = [X ; ones(1,size(X,2))]; % Add a bais term to all the data points
    z2 = weight_matrix_1 * a1;
    a2 = [sigmoid(z2) ; ones(1,size(z2,2))];
    z3 = weight_matrix_2 * a2;
    a3 = [sigmoid(z3) ; ones(1,size(z3,2))];
    z4 = weight_matrix_3 * a3;
    a4 = [sigmoid(z4) ; ones(1,size(z4,2))];
    z5 = weight_matrix_4 * a4;
    a5 = softmax(z5);

    %% Backward Propogation
    d5 = a5 - Y;
    d4 = (weight_matrix_4' * d5).* [sigmoidGradient(z4) ; ones(1,size(z4,2))];
    d3 = (weight_matrix_3' * d4) .* [sigmoidGradient(z3) ; ones(1,size(z3,2))];
    d2 = (weight_matrix_2' * d3) .* [sigmoidGradient(z2) ; ones(1,size(z2,2))];
    
    %% Calculate the Gradient
    grad1 = 


end

