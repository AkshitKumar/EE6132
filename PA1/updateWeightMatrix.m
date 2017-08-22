function [ wm1,wm2,wm3,wm4 ] = updateWeightMatrix(wm1,wg1,wm2,wg2,wm3,wg3,wm4,wg4,step_size)
    wm1 = wm1 - (step_size * wg1);
    wm2 = wm2 - (step_size * wg2);
    wm3 = wm3 - (step_size * wg3);
    wm4 = wm4 - (step_size * wg4);
end

