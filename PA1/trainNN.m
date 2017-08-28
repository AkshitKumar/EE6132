function [training_loss,test_loss,test_acc] = trainNN(num_iter,train_images,train_labels,test_images,test_labels,activation,lambda,alpha,weight_matrix_1,weight_matrix_2,weight_matrix_3,weight_matrix_4,alpha_decay,output_layer_size,velocity_1,velocity_2,velocity_3,velocity_4)
    training_loss = [];
    test_loss = [];
    test_acc = [];
    for i = 1:num_iter
        r = randi([1 size(train_images,2)],1,64);
        X = train_images(:,r);
        y = train_labels(r,:);
        [train_loss,weight_grad_1 , weight_grad_2 , weight_grad_3 , weight_grad_4] = calculateGradient(X',y,activation,output_layer_size,lambda,weight_matrix_1,weight_matrix_2, weight_matrix_3, weight_matrix_4); 
        if(alpha_decay == 1 && mod(i,250) == 0)
            alpha = 0.85 * alpha; % Decay the alpha
        end
        [weight_matrix_1, weight_matrix_2 , weight_matrix_3, weight_matrix_4,velocity_1,velocity_2,velocity_3,velocity_4] = updateWeightMatrix(weight_matrix_1,weight_grad_1,weight_matrix_2,weight_grad_2,weight_matrix_3,weight_grad_3,weight_matrix_4,weight_grad_4,alpha,velocity_1,velocity_2,velocity_3,velocity_4);
        training_loss = [training_loss ; train_loss];
        if mod(i,200) == 0
            [loss, acc] = calculateTestLoss(test_images',test_labels,output_layer_size,activation,lambda,weight_matrix_1,weight_matrix_2,weight_matrix_3,weight_matrix_4);
            test_loss = [test_loss ; loss];
            test_acc = [test_acc; acc];
        end
    end
end

