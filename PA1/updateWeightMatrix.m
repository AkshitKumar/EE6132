function [ wm1,wm2,wm3,wm4,v1,v2,v3,v4] = updateWeightMatrix(wm1,wg1,wm2,wg2,wm3,wg3,wm4,wg4,step_size,v1,v2,v3,v4)
    alpha = 0.5; %% Hardcoding the parameter
    %% Update Velocities
    v1 = alpha * v1 - step_size * wg1;
    v2 = alpha * v2 - step_size * wg2;
    v3 = alpha * v3 - step_size * wg3;
    v4 = alpha * v4 - step_size * wg4;
    %% Update the weights
    wm1 = wm1 + v1;
    wm2 = wm2 + v2;
    wm3 = wm3 + v3;
    wm4 = wm4 + v4;
end

