function [estimate_value, estimates] = getTopThreeEstimates(learnt_weight_1, learnt_weight_2 , learnt_weight_3, learnt_weight_4, activation,test_images)
    X = test_images(:,1:20);
    X = X';
    %% Do forward Propagation
    if activation == 0
        a1 = [ones(size(X,1),1) X]; % Add a bias term
        z2 = a1 * learnt_weight_1' ;
        a2 = [ones(size(z2,1),1) sigmoid(z2)];
        z3 = a2 * learnt_weight_2';
        a3 = [ones(size(z3,1),1) sigmoid(z3)];
        z4 = a3 * learnt_weight_3';
        a4 = [ones(size(z4,1),1) sigmoid(z4)];
        z5 = a4 * learnt_weight_4';
        a5 = softmax(z5')';
    else 
        a1 = [ones(size(X,1),1) X]; % Add a bias term
        z2 = a1 * learnt_weight_1' ;
        a2 = [ones(size(z2,1),1) relu(z2)];
        z3 = a2 * learnt_weight_2';
        a3 = [ones(size(z3,1),1) relu(z3)];
        z4 = a3 * learnt_weight_3';
        a4 = [ones(size(z4,1),1) relu(z4)];
        z5 = a4 * learnt_weight_4';
        a5 = softmax(z5')';
    end
    [B,index] = sort(a5,2,'descend');
    estimates = index(:,1:3) - 1;
    estimate_value = B(:,1:3);
end

