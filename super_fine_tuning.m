function [deltaError,delta] = super_fine_tuning(stage,regParams,top2label,x,y,deltaError_next,isTop)
%Function£º
%        fine tune the regression parameters
%
%Input£º
%               stage - stage index
%           regParams - regression parameters
%           top2label - regression weight for classification
%                   x - previous stage representation
%                   y - current stage representation at last time step
%     deltaError_next - BP error signal from next stage
%               isTop - if the stage is the top stage of feature extraction
%
%Output£º
%      deltaError - current stage BP error signal
%           delta - negtive gradient


if isTop
    W_next = top2label;
else
    W_next  = regParams.W{stage+1};
end

deriv_actFun = y.*(1-y);

%BP error signal
deltaError = (W_next'*deltaError_next).*deriv_actFun;

delta.W = deltaError*x';
delta.b_r = deltaError;


