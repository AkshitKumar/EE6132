function weight_matrix = randWeightInit(L_in, L_out)
    dimension = [L_out , L_in + 1];
    weight_matrix = random('normal',0,0.08,dimension);
end

