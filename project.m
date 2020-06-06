

close all; clc; clear all
digitDatasetPath='C:\Users\erika\Desktop\CNN\PennFudanPed\PNGImages'; % set folder where image folders are
% make subfolders for each category

imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames'); % collects all images in subfolders

nfiles = length(imds.Files); %number of files in full dataset
labelCount = countEachLabel(imds); % labels based on subfolders
nlabel = length(labelCount{:,2}); % number of folders

% Display a random selection of images to check
figure;% figure for showing images
perm = randperm(round(nfiles/2),20);
for i = 1:20
    subplot(4,5,i);
    imshow(imds.Files{perm(i)});
end

% Display categories
disp(labelCount)
fprintf('Total: %g\n',nfiles)

% Set image size pixel count
img = readimage(imds,1);  % reads first image in set
imgsize = size(img); %sets size of all images based on first in set
imgsize = [28,28, 3]; % overrides size of all images
fprintf('Image size used: %g %g %g\n',imgsize);

[imdsTrain,imdsValidation] = splitEachLabel(imds,0.5,'randomized'); % 50% of images are training
auimdsTrain = augmentedImageDatastore(imgsize,imdsTrain); % resizes Training images
auimdsValidation = augmentedImageDatastore(imgsize,imdsValidation); % resizes validation images

% set layers
stridepool=2;
layers = [
    imageInputLayer(imgsize)
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',stridepool)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',stridepool)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    %         fullyConnectedLayer(2)
    fullyConnectedLayer(nlabel)
    %     fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

% Set options for training
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',auimdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');
disp(options);
disp(layers)
t1=tic;
net = trainNetwork(auimdsTrain,layers,options); % runs simulation
dt=toc(t1);

% dt=etime(t1,t2);
fprintf('Time to train  %g\n',dt);

YPred = classify(net,auimdsValidation);
YValidation = imdsValidation.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation);
fprintf('Accuracy: %g\n',accuracy);

