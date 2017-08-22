function rGrad = reluGradient(z)
    rGrad = max(z,0);
    rGrad(find(rGrad > 0)) = 1;
end