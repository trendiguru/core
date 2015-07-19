%__author__ = 'jeremy'
%this is a matlab function
%which takes an image filename
%and returns a parse mask (integers represneting classes)
%using paperdoll

function label_mask = pd(image_filename)
%todo - check if path already ok, 
% check if data already loaded

load data/paperdoll_pipeline.mat config;
addpath(genpath('.'))
input_image = imread(image_filename);
input_sample = struct('image', imencode(input_image, 'jpg'));
config{1}.scale = 200;  
config{1}.model.thresh = -2;   

result = feature_calculator.apply(config, input_sample)

label_mask = imdecode(result.final_labeling, 'png');
imwrite(label_mask,'output.jpg')
save('labels.mat','label_mask')
%show_parsing(result.image, result.final_labeling, result.refined_lab
els);
save('output.mat','result')
return
