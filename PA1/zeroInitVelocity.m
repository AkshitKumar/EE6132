function [ velocity ] = zeroInitVelocity(L_in,L_out)
    dimension = [L_out, L_in + 1];
    velocity = zeros(dimension);
end

