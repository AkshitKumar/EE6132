function sigGrad = sigmoidGradient(z)
    sigGrad = sigmoid(z).*(1-sigmoid(z));
end

