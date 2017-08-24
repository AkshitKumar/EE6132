function [L,acc] = calculateTestLoss(X,y,num_labels,lambda,weight_matrix_1,weight_matrix_2,weight_matrix_3,weight_matrix_4)
    %% Number of data points
    m = size(X,1);

    %% Convert y into one hot encoding
    p = eye(num_labels);
    Y = p(y+1,:);
    
    %% Forward Propogation
    a1 = [ones(size(X,1),1) X]; % Add a bias term 
    z2 = a1 * weight_matrix_1' ; 
    a2 = [ones(size(z2,1),1) sigmoid(z2)];
    z3 = a2 * weight_matrix_2';
    a3 = [ones(size(z3,1),1) sigmoid(z3)];
    z4 = a3 * weight_matrix_3';
    a4 = [ones(size(z4,1),1) sigmoid(z4)];
    z5 = a4 * weight_matrix_4';
    a5 = softmax(z5')';

    %% Calculate the loss function
    L = -(sum(sum(Y.*log(a5))));
    L = L/m;
    reg = (lambda/(2*m)) * (sum(sum(weight_matrix_1(:,2:end).^2)) + sum(sum(weight_matrix_2(:,2:end).^2)) + sum(sum(weight_matrix_3(:,2:end).^2)) + ... 
          sum(sum(weight_matrix_4(:,2:end).^2)));
    L = L + reg;
    
    %% Calculate the test accuracy
    [m,index] = max(a5,[],2);
    matches = (index == y+1);
    acc = sum(matches) / length(y);

end

