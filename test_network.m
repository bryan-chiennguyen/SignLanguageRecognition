clear all;  %delete param
close all;  %close all window
clc;        %clear all command
net = load('net_Pre_goo_biLstm.mat');
net_sign = net.net_scenario;  
total = numel(net.imdsTest.Files);
right = 0;
accuracy_temp = 100;
labels_pred = [];
for i = 1:numel(net.imdsTest.Files)
    fprintf('\n reading video %d in total %d video',i,total);
    filename = net.imdsTest.Files(i);
    video = sliceWindown_func_map(char(filename));
    numFrames = size(video,4);
    %predict video

    Ypred = classify(net_sign,{video});
    labels_pred = [labels_pred;Ypred];
    fprintf('      sign name %s',Ypred);
    if(Ypred == net.imdsTest.Labels(i))
        right = right + 1;
    else
        accuracy_temp = accuracy_temp - (1/total)*100;   
    end
     fprintf('          accuracy temp %3.2f \n',accuracy_temp);
end
accuracy = (right/total)*100;
% plotconfusion(net.imdsTest.Labels,labels_pred);
confusionchart(net.imdsTest.Labels,labels_pred);