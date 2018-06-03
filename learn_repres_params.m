function [y_new,regParams,genParams] = learn_repres_params(stage,regParams,genParams,x,y,y_td_next,learn_rates,alpha,beta,lr_decay,momentum) 
%Function:
%        learn representation and parameters for one feature extraction stage
%
%Input:
%            stage - stage index
%        regParams - regression parameters
%        genParams - generative parameters
%                x - previous stage representation
%                y - current stage representation at last time step
%        y_td_next - top-down prediction from next stage
%      learn_rates - learning rates
%            alpha - trade-off parameter
%             beta - trade-off parameter
%         lr_decay - learning decay coefficient
%         momentum - momentum coefficient
%
%Output:
%          y_new - updated current stage representation
%      regParams - updated regression parameters
%      genParams - updated generative parameters


W_old   = regParams.W{stage};
b_r_old = regParams.b_r{stage};
V_old   = genParams.V{stage};
b_g_old = genParams.b_g{stage};

mom_V   = genParams.mom_V{stage};
mom_b_g = genParams.mom_b_g{stage};
mom_W   = regParams.mom_W{stage};
mom_b_r = regParams.mom_b_r{stage};

lr_y   = learn_rates.y;
lr_W   = learn_rates.W/lr_decay;
lr_b_r = learn_rates.b_r/lr_decay;
lr_V   = learn_rates.V/lr_decay;
lr_b_g = learn_rates.b_g/lr_decay;

lambda_y = 0;
lambda_r = 0;
lambda_g = 0;

[n, ~] = size(W_old);
if isempty(y)
    y = zeros(n, 1, 'single');
end
if isempty(y_td_next)
    y_td_next = zeros(size(y));
end


%sigmoid regression and generative functions
G_y2x = sigmoid(V_old*y + b_g_old);
publicTerm_G = (x - G_y2x).*G_y2x.*(1-G_y2x);
R_x2y = sigmoid(W_old*x + b_r_old);
publicTerm_R = (y - R_x2y).*R_x2y.*(1-R_x2y);

%update representation
increment_y = lr_y*(V_old'*publicTerm_G - alpha*(y-y_td_next) - beta*(y-R_x2y) -lambda_y*y);
y_new = y + increment_y;

if nargout > 1 
    %update regression parameters
    increment_W = lr_W*(beta*publicTerm_R*x' - lambda_r*W_old);
    mom_W = -momentum*mom_W + increment_W;
    W_new = W_old + mom_W;
    increment_b_r = lr_b_r*(beta*publicTerm_R);
    mom_b_r = -momentum*mom_b_r + increment_b_r;
    b_r_new = b_r_old + mom_b_r;

    %update generative parameters
    increment_V = lr_V*(publicTerm_G*y' - lambda_g*V_old);
    mom_V = -momentum*mom_V + increment_V;
    V_new = V_old + mom_V;
    increment_b_g = lr_b_g*publicTerm_G;
    mom_b_g = -momentum*mom_b_g + increment_b_g;
    b_g_new = b_g_old + mom_b_g;

else
    W_new   = W_old;
    b_r_new = b_r_old;
    V_new   = V_old;
    b_g_new = b_g_old;
end

regParams.W{stage}   = W_new;
regParams.b_r{stage} = b_r_new;
genParams.V{stage}   = V_new;
genParams.b_g{stage} = b_g_new;

genParams.mom_V{stage}   = mom_V;
genParams.mom_b_g{stage} = mom_b_g;
regParams.mom_W{stage}   = mom_W;
regParams.mom_b_r{stage} = mom_b_r;


