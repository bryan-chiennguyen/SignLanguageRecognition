% this file is for constructing and training network 

clear all;
close all;
clc;

net = googlenet;        %CNN net

imds = imageDatastore('Data','IncludeSubfolders',true,'LabelSource','foldernames','FileExtensions','.avi'); % data path
ratio_init = 0.6;   % train/test = 6/4
[imdsTrain,imdsTest] = splitEachLabel(imds,ratio_init,'randomized');    % using imdsTrain for training

labels = imdsTrain.Labels;  % save labels for training
numFiles = numel(imdsTrain.Files);  
%==============Augment Data train===========
inputSize = net.Layers(1).InputSize(1:2);

augmenter = imageDataAugmenter('RandYScale',[1 2],'RandXScale',[1 2]); %'RandRotation',[-10 10]
augNumber = 1;      % don't augment data!



%==============Feature extraction============
% save processed video in temporary folder, 
% this code below build for basic classify (availble)

layerName = "pool5-7x7_s1";
tempFile = fullfile(tempdir,"signLang_Pre_map_googlenet.mat");
if exist(tempFile,'file')
    load(tempFile,"sequences");
    if (augNumber>1)
    for i = 1:(augNumber-1)
    labels = [labels;labels];
    end
    end
    
else
            sequences = cell(numFiles*augNumber,1);   % Cell can contain whole type of data
    for k = 1:augNumber
        if(k==1)
            fprintf("Change to sequences without augmentation \n");
            for i = 1:numFiles
                    fprintf("Reading file %d of %d...\n", i, numFiles)
                    video = sliceWindown_func_map(char(imdsTrain.Files(i)));    % feature extractor 
                     sequences{((k-1)*numFiles)+i,1} = activations(net,video,layerName,'OutputAs','columns');    %sequences for 1 video
             end
        else
            fprintf("Augmenting %d....\n",k);
                for i = 1:numFiles
                    fprintf("Reading file %d of %d...\n", i, numFiles)
                    video = readVideo(char(imdsTrain.Files(i)));
                    video = sliceWindown_func_map(char(imdsTrain.Files(i)));
                    video = augment_video(video,augmenter);
                     sequences{((k-1)*numFiles)+i,1} = activations(net,video,layerName,'OutputAs','columns');    %sequences for 1 video
                    labels = [labels;labels(i)];
                end
        end
    end
       
    save(tempFile,"sequences","-v7.3");
end

%%
%============= divide sequences to train and validation=========
numObservations = numel(sequences);
idx = randperm(numObservations);    % creat array contain permutation of total video

numTrain = floor(0.8*numObservations); % ratio of train/validation is 9/1 
idxTrain = idx(1:numTrain);
sequencesTrain = sequences(idxTrain);   % sequences of video in train data
labelsTrain = labels(idxTrain);

idxValidation = idx(numTrain+1:end);
sequencesValidation = sequences(idxValidation);
labelValidation = labels(idxValidation);

numclasses = numel(categories(labelsTrain));    % number sign
numFeatures = size(sequencesTrain{1},1);        %number feature in frames (need to optimize)

% creat BiLSTM network and connect to googleNet
layers = [
  sequenceInputLayer(numFeatures,'Name','sequence')
  bilstmLayer(1024, 'OutputMode','last','Name','bilstm')    
  dropoutLayer(0.5,'Name','dropout')
  fullyConnectedLayer(numclasses,'Name','fc')
  softmaxLayer('Name','sofmax')
  classificationLayer('Name','classification')
  
];

%=============== specify training option=========

miniBatchSize =32;
numIterationsPerEpoch = floor(numObservations/miniBatchSize);
options = trainingOptions('adam',...
    'MaxEpochs',15,...
    'MiniBatchSize',miniBatchSize,...
    'InitialLearnRate',1e-4,...
    'GradientThreshold',2,...
    'Shuffle','every-epoch',...
    'ValidationData',{sequencesValidation,labelValidation},...
    'ValidationFrequency',numIterationsPerEpoch,...
    'ValidationPatience',2,...            
    'Plots','training-progress',...
    'Verbose',false);

%===========train network============
[netLSTM,info] = trainNetwork(sequencesTrain,labelsTrain,layers,options);

%=============Assemble network========
% this work can handle by deep neural network design
cnnLayers = layerGraph(net);
%delete unused layer
layerNames = ["data" "pool5-drop_7x7_s1" "loss3-classifier" "prob" "output"];
cnnLayers = removeLayers(cnnLayers,layerNames);
% add sequences input layer for CNN
inputSize = net.Layers(1).InputSize(1:2);
averageImage = net.Layers(1).AverageImage;
inputLayer = sequenceInputLayer([inputSize 3], ...
    'Name','input',...
    'Normalization','zerocenter',...
    'Mean',averageImage);

layers = [
    inputLayer
    sequenceFoldingLayer('Name','fold')];

lgraph = addLayers(cnnLayers,layers);
lgraph = connectLayers(lgraph,"fold/out","conv1-7x7_s2");

%Add BiLSTM layers to 
lstmLayers = netLSTM.Layers;
lstmLayers(1) = [];

layers = [
    sequenceUnfoldingLayer('Name','unfold')
    flattenLayer('Name','flatten')
    lstmLayers];

lgraph = addLayers(lgraph,layers);
lgraph = connectLayers(lgraph,"pool5-7x7_s1","unfold/in");

lgraph = connectLayers(lgraph,"fold/miniBatchSize","unfold/miniBatchSize");

net_scenario = assembleNetwork(lgraph);
save('net_Pre_goo_BiLstm.mat');         

%run_test;
test_network;

