function [regParams, genParams, classRegParams, classGenParams] = FIPC(data_train, labelTrain, alpha, beta)
%Function:
%        Learn model parameters
%
%Input:
%     data_train - training data (num x dim)
%     labelTrain - label (0-1 matrix: num x 10)
%          alpha - trade-off parameter
%           beta - trade-off parameter
%
%Output:
%           regParams - regression parameters for feature extraction model
%           genParams - generative parameters for feature extraction model
%      classRegParams - regression parameters for classification model
%      classGenParams - generative parameters for classification model


%====================== 0 set initial parameters ==========================
global numStages nr npr
[numTrain, dim] = size(data_train);
batch_size = 100;
numBatch = numTrain/batch_size;
epochs_PC = 10;
show = 1;
iterations_PC = 10;
epochs_BP = 20;

momentum_PC = 0.5;
lr_decay_PC = 1;
lr_decay_class = 1;
momentum_BP = 0;
lr_BP_featureExtract = 10;
lr_BP_class = 5;


%===================== 1 initialize weights and biases =====================
for stage = 1 : numStages
    
    [param_reg, param_gen] = weight_initialisation_random(npr(stage+1)*nr(stage+1), npr(stage)*nr(stage));
    regParams.W{stage}   = param_reg(:, 2:end);
    regParams.b_r{stage} = param_reg(:, 1);
    genParams.V{stage}   = param_gen(:, 2:end);
    genParams.b_g{stage} = param_gen(:, 1);
    
    genParams.mom_V{stage} = zeros(size(genParams.V{stage}));
    genParams.mom_b_g{stage} = zeros(size(genParams.b_g{stage}));
    regParams.mom_W{stage} = zeros(size(regParams.W{stage}));
    regParams.mom_b_r{stage} = zeros(size(regParams.b_r{stage}));

    %localize weights
    ranges{stage} = split_range(nr(stage), nr(stage+1), 1);  
    for k = 1 : nr(stage+1)  
        NodesInRegion(k,:) = ((k-1)*npr(stage+1)+1) : (k*npr(stage+1));
    end
    [regParams.W{stage}, genParams.V{stage}, weight_localID{stage}] = restrict_RF(regParams.W{stage}, genParams.V{stage}, NodesInRegion,...
								                       nr(stage), npr(stage), ranges{stage});   
    clear NodesInRegion;

end
[param_top2label,param_label2top] = weight_initialisation_random(size(labelTrain,2), npr(numStages+1)*nr(numStages+1));
classRegParams.top2label = param_top2label(:, 2:end);
classRegParams.b_lab = param_top2label(:, 1);
classGenParams.label2top = param_label2top(:, 2:end);
classGenParams.b_top = param_label2top(:, 1);

classRegParams.mom_top2label = zeros(size(classRegParams.top2label));
classRegParams.mom_b_lab     = zeros(size(classRegParams.b_lab));
classGenParams.mom_label2top = zeros(size(classGenParams.label2top));
classGenParams.mom_b_top     = zeros(size(classGenParams.b_top));

%initialize representations and learning rates
for stage = 1 : numStages+1
    ymax{stage} = 0;
    y{stage}    = [];
    y_td{stage} = [];
    learn_rates{stage}.y   = 0.1;
    learn_rates{stage}.W   = 0.05;
    learn_rates{stage}.b_r = 0.05;
    learn_rates{stage}.V   = 0.005;
    learn_rates{stage}.b_g = 0.005;
end
lr_classReg = 0.005;
lr_classGen = 0.005;


%===================== 2 pre-train model ===============================
for k = 1 : epochs_PC

    rand_index = randperm(numTrain);
    for batch = 1 : numBatch        
        
        batch_data  = data_train(rand_index((batch-1)*batch_size+1:batch*batch_size), :);
        batch_label = labelTrain(rand_index((batch-1)*batch_size+1:batch*batch_size), :);
        
        for i = 1 : batch_size
            x{1}  = batch_data(i, :)';
            label = batch_label(i, :)';
            
            %initialize each stage's representation
            for stage = 1 : numStages        
                y{stage} = [];       
                y_td{stage+1} = [];
            end
            
            for t = 1 : iterations_PC

                %feature extraction sub-model
                for stage = 1 : numStages
                    [y{stage},regParams,genParams] = learn_repres_params(stage,regParams,genParams,x{stage},y{stage},...
								        y_td{stage+1},learn_rates{stage},alpha,beta,lr_decay_PC,momentum_PC);
                    %localize weights
                    regParams.W{stage} = regParams.W{stage}.*weight_localID{stage};
                    genParams.V{stage} = weight_localID{stage}'.*genParams.V{stage};   
                    %update prediction for previous stage
                    y_td{stage} = genParams.V{stage}*y{stage} + genParams.b_g{stage};
                    y_td{stage} = sigmoid(y_td{stage});
                    
                    x{stage+1} = y{stage}; 
                end
                
                %classification sub-model
                [label_output, classRegParams, classGenParams] = learn_label(y{numStages},label,classRegParams,classGenParams,lr_classReg,lr_classGen,lr_decay_class,momentum_PC);
                %update prediction for previous stage
                y_td{numStages+1} = classGenParams.label2top*label_output + classGenParams.b_top;
                y_td{numStages+1} = sigmoid(y_td{numStages+1});
            end 
        end
    end
    
    %display results during training
    for stage = 1 : numStages
        ymax{stage} = max([ymax{stage}, max(y{stage})]);
    end
    if rem(k, show) == 0
        fprintf(1,'----------pre_training: epoch %i...\n',k);
        for stage = 1 : numStages
            disp(['STAGE ',int2str(stage),': ymax=',num2str(ymax{stage}),...
			' wSum=',num2str(max(sum(regParams.W{stage},2))),...
			' vSum=',num2str(max(sum(genParams.V{stage})))]);
            ymax{stage} = 0; 
        end
        disp(['top2labelSum = ',num2str(max(sum(classRegParams.top2label,2)))]);
    end
end
save('Results\FIPC_afterPreTrain.mat','classGenParams','classRegParams','genParams', 'regParams');


%===================== 3 fine-tune model ==================================
for stage = 1 : numStages
    ymax{stage} = 0;
end
      
for stage = 1 : numStages
    regParams.mom_W{stage}   = zeros(size(regParams.W{stage}));
    regParams.mom_b_r{stage} = zeros(size(regParams.b_r{stage}));
end
classRegParams.mom_top2label = zeros(size(classRegParams.top2label));        
classRegParams.mom_b_lab     = zeros(size(classRegParams.b_lab));

for k = 1 : epochs_BP
    
    rand_index = randperm(numTrain);
    for batch = 1 : numBatch
        for stage = 1 : numStages
            sum_delta.W{stage}    = zeros(size(regParams.W{stage}));
            sum_delta.b_r{stage} = zeros(size(regParams.b_r{stage}));
        end
        sum_delta.top2label = zeros(size(classRegParams.top2label));
        sum_delta.b_lab     = zeros(size(classRegParams.b_lab));

        batch_data  = data_train(rand_index((batch-1)*batch_size+1:batch*batch_size), :);
        batch_label = labelTrain(rand_index((batch-1)*batch_size+1:batch*batch_size), :);
        for i = 1 : batch_size
            x{1}  = batch_data(i, :)';
            label = batch_label(i, :)';
            
            %compute representation in a bottom-up manner
            for stage = 1 : numStages
                y{stage} = regParams.W{stage}*x{stage} + regParams.b_r{stage};
                y{stage} = sigmoid(y{stage});
                x{stage+1} = y{stage};
            
            end
            labe_input = classRegParams.top2label*y{numStages} + classRegParams.b_lab;
            
            label_output = sigmoid(labe_input);
               
            %propagate error in a top-down manner
            deltaError{numStages+1} = (label - label_output).*label_output.*(1-label_output);
            delta.top2label     = deltaError{numStages+1}*y{numStages}';
            sum_delta.top2label = sum_delta.top2label + delta.top2label;
            delta.b_lab     = deltaError{numStages+1};
            sum_delta.b_lab = sum_delta.b_lab + delta.b_lab;
            for stage = numStages : -1 : 1
                isTop = 0;
                if stage == numStages
                    isTop = 1;
                end
                [deltaError{stage},delta] = super_fine_tuning(stage,regParams,classRegParams.top2label,x{stage},y{stage},deltaError{stage+1},isTop);
                sum_delta.W{stage}   = sum_delta.W{stage} + delta.W;
                sum_delta.b_r{stage} = sum_delta.b_r{stage} + delta.b_r;
            end
        end
        
        %update regression parameters for feature extraction sub-model
        for stage = 1 : numStages
            increment_W        = lr_BP_featureExtract/batch_size*sum_delta.W{stage};
            regParams.mom_W{stage} = -momentum_BP*regParams.mom_W{stage} + increment_W;
            regParams.W{stage} = regParams.W{stage} + regParams.mom_W{stage};
            regParams.W{stage} = regParams.W{stage}.*weight_localID{stage};
            
            increment_b_r            = lr_BP_featureExtract/batch_size*sum_delta.b_r{stage};
            regParams.mom_b_r{stage} = -momentum_BP*regParams.mom_b_r{stage} + increment_b_r;
            regParams.b_r{stage}     = regParams.b_r{stage} + regParams.mom_b_r{stage};

            ymax{stage} = max([ymax{stage}, max(y{stage})]);
        end
        %update regression parameters for classification sub-model
        increment_top2label      = lr_BP_class/batch_size*sum_delta.top2label;
        classRegParams.mom_top2label = -momentum_BP*classRegParams.mom_top2label + increment_top2label;
        classRegParams.top2label = classRegParams.top2label + classRegParams.mom_top2label;
        increment_b_lab      = lr_BP_class/batch_size*sum_delta.b_lab;
        classRegParams.mom_b_lab = -momentum_BP*classRegParams.mom_b_lab + increment_b_lab;
        classRegParams.b_lab = classRegParams.b_lab + classRegParams.mom_b_lab;
        
    end
    
    %display results during training
    if rem(k, show) == 0
        fprintf(1,'----------fine_tuning: epoch %i...\n',k);
        for stage = 1 : numStages
            disp(['STAGE ',int2str(stage),': wSum=',num2str(max(sum(regParams.W{stage},2)))]);
            ymax{stage} = 0; 
        end
        disp(['top2labelSum = ',num2str(max(sum(classRegParams.top2label,2)))]);
    end
end
save('Results\FIPC_afterFineTune.mat','classGenParams','classRegParams','genParams', 'regParams');

        
