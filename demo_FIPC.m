%Function: Using Fast Inference Predictive Coding (FIPC) Model to solve the
%image classification task on MNSIT dataset


rand('state', 0);
randn('state', 0);


%========================== 0 set initial parameters ======================
global numStages nr npr
numStages = 1;
img_width = 28;
nr = [img_width^2, 4];
npr = [1, 25];


%========================== 1 preprocess data =============================
load('MNIST_28x28.mat');
data_train_all  = images_train';
label_train_all = labels_train;
data_test   = images_test';
label_test  = labels_test;

numClass = 10;
labelTrain_all = zeros(length(label_train_all), numClass);
labelTest  = zeros(length(label_test), numClass);
for class = 1 : numClass
    %training data label 
    ind_train_all = (label_train_all == (class-1));
    labelTrain_all(ind_train_all, class) = 1;
    %test data label
    ind_test = (label_test == (class-1));
    labelTest(ind_test, class) = 1;
end
%obtain the validation set and new training set from the original training set
numTrain = 5000;
data_train = data_train_all(1:numTrain, :);
labelTrain = labelTrain_all(1:numTrain, :);
numValid = 10000;
data_valid = data_train_all(50001:end, :);
labelValid = labelTrain_all(50001:end, :);
numTest = size(data_test, 1);


%========================== 2 learn model parameters ======================
alpha = 0.4;
beta  = 1;
[regParams, genParams, classRegParams, classGenParams] = FIPC(data_train, labelTrain, alpha, beta);


%========================== 3 test model on training set ==================
error_train = 0;
for data_id = 1 : numTrain
    img_train = data_train(data_id, :)';
    label_data = labelTrain(data_id, :)';
    [~, label] = max(label_data);
    [~, ~, class_pre] = inference(img_train, label_data, regParams, genParams, classRegParams, classGenParams);    
    if class_pre ~= label    
        error_train = error_train + 1;
    end
end
FIPC_errorRate_train = error_train/numTrain*100;


%========================== 4 test model on validation set ================
error_valid = 0;
for data_id = 1 : numValid
    img_valid = data_valid(data_id, :)';
    label_data = labelValid(data_id, :)';
    [~, label] = max(label_data) ;
    [~, ~, class_pre] = inference(img_valid, label_data, regParams, genParams, classRegParams, classGenParams);
    if class_pre ~= label
        error_valid = error_valid + 1;
    end
end
FIPC_errorRate_valid = error_valid/numValid*100;


%========================== 5 test model on test set ======================
error_test = 0;
for data_id = 1 : numTest
    img_test = data_test(data_id, :)';
    label_data = labelTest(data_id, :)';
    [~, label] = max(label_data) ;
    [~, ~, class_pre] = inference(img_test, label_data, regParams, genParams, classRegParams, classGenParams);
    if class_pre ~= label
        error_test = error_test + 1;
    end
end
FIPC_errorRate_test = error_test/numTest*100;






