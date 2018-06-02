function [param_reg, param_gen] = weight_initialisation_random(n, m)

%regression parameters
param_reg = (rand(n, m+1) - 0.5) * 2 * 4 * sqrt(6 / (n + m));

%generative parameters
if nargout > 1
    param_gen = (rand(m, n+1) - 0.5) * 2 * 4 * sqrt(6 / (m + n));
end









