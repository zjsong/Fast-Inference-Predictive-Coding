function [label_output,classRegParams,classGenParams] = learn_label(y_top,label,classRegParams,classGenParams,lr_R,lr_G,lr_decay,momentum)
%Function:
%        learn representation and parameters for classification sub-modle
%
%Input:
%               y_top - representation of the top feature extraction stage
%               label - data label (one-hot vector)
%      classRegParams - regression parameters
%      classGenParams - generative parameters
%                lr_R - learning rate of regression parameters
%                lr_G - learning rate of generative parameters
%            lr_decay - learning rate decay coefficient
%            momentum - momentum coefficient
%
%Output:
%        label_output - representation of classification stage
%      classRegParams - updated regression parameters
%      classGenParams - updated generative parameters


top2label = classRegParams.top2label;
b_lab     = classRegParams.b_lab;
label2top = classGenParams.label2top;
b_top     = classGenParams.b_top;

mom_top2label = classRegParams.mom_top2label;
mom_b_lab     = classRegParams.mom_b_lab;
mom_label2top = classGenParams.mom_label2top;
mom_b_top     = classGenParams.mom_b_top;

lr_R = lr_R/lr_decay;
lr_G = lr_G/lr_decay;

lambda = 0;

label_output = sigmoid(top2label*y_top + b_lab);
publicTerm_2label = (label - label_output).*label_output.*(1 - label_output);

y_top_gen = sigmoid(label2top*label_output + b_top);
publicTerm_2top = (y_top - y_top_gen).*y_top_gen.*(1 - y_top_gen);

grad_top2label = -publicTerm_2label*y_top';
mom_top2label = -momentum*mom_top2label - lr_R*grad_top2label - lambda*top2label;
top2label = top2label + mom_top2label;
grad_b_lab = -publicTerm_2label;
mom_b_lab = -momentum*mom_b_lab - lr_R*grad_b_lab;
b_lab = b_lab + mom_b_lab;

label_output = sigmoid(top2label*y_top + b_lab);

grad_label2top = -publicTerm_2top*label_output';
mom_label2top = -momentum*mom_label2top - lr_G*grad_label2top - lambda*label2top;
label2top = label2top + mom_label2top;
grad_b_top = -publicTerm_2top;
mom_b_top = -momentum*mom_b_top - lr_G*grad_b_top;
b_top = b_top + mom_b_top;

classRegParams.top2label = top2label;
classRegParams.b_lab     = b_lab;
classGenParams.label2top = label2top;
classGenParams.b_top     = b_top;

classRegParams.mom_top2label = mom_top2label;
classRegParams.mom_b_lab     = mom_b_lab;
classGenParams.mom_label2top = mom_label2top;
classGenParams.mom_b_top     = mom_b_top;


