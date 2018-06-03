function [img_recon, y, class_pre] = inference(img, label, regParams, genParams, classRecParams, classGenParams)
%Function:
%        reconstruct input image
%
%Input:
%                 img - input image(column vector)
%               label - label of the input image (one-hot column vector)
%           regParams - regression parameters for feature extraction model
%           genParams - generative parameters for feature extraction model
%      classRegParams - regression parameters for classification model
%      classGenParams - generative parameters for classification model
%
%Output:
%      img_recon - reconstructed image
%              y - representations for all feature extraction stages
%      class_pre - predicted label


W   = regParams.W;
b_r = regParams.b_r;
V   = genParams.V;
b_g = genParams.b_g;

top2label = classRecParams.top2label;
b_lab     = classRecParams.b_lab;
label2top = classGenParams.label2top;
b_top     = classGenParams.b_top;

width = sqrt(length(img));
[~, numStages] = size(W);
img_recon = zeros(width, width, numStages);


%predict label
x{1} = img;
for stage = 1 : numStages
    y{stage} = W{stage}*x{stage} + b_r{stage};
    y{stage} = sigmoid(y{stage});
    x{stage+1} = y{stage};
end
y_pre = sigmoid(top2label*y{numStages} + b_lab);
[~, class_pre] = max(y_pre);

% %reconstruct image
% for stage = 1 :  numStages
%     recon_ss = y{stage};
%     for ss = stage : -1 : 1
%         recon_ss = V{ss}*recon_ss + b_g{ss};
%         recon_ss = sigmoid(recon_ss);        
%     end
%     img_recon(:, :, stage) = reshape(recon_ss, [width, width]);
% end    
% 
% Img = reshape(img, [width,width]);
% figure(20),clf,
% imshow(Img, []);
% axis('equal','tight'); set(gca,'XTick',[],'YTick',[]);
% for stage = 1 : numStages
%     figure(stage+20),clf,
%     imshow(img_recon(:,:,stage), []);
%     axis('equal','tight'); set(gca,'XTick',[],'YTick',[]);
% end


