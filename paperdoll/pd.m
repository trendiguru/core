function [mask,label_names,pose] = pd(image_filename)
%todo - check if path already ok,
% check if data already loaded
disp(['the image sent to pd in matlab is:' image_filename])
%todo - check if we cant load this once only (when engine is created)

%profile on
tic

mask=[];
label_names = [];
pose = [];
load data/paperdoll_pipeline.mat config;
addpath(genpath('.'))
input_image = imread(image_filename);
input_sample = struct('image', imencode(input_image, 'jpg'));
config{1}.scale = 200;
config{1}.model.thresh = -2;

result = feature_calculator.apply(config, input_sample)
if  ~isfield(result, 'final_labeling')
    % paperdoll failed to return result
    disp('XXXXXXXXXXisfield switch enteredXXXXXXX');
    failname = strcat('home/jeremy/pd_output/fail.',image_filename)
    disp(['failfile name' failname] )
    imwrite(input_image,failname)
    return
end

mask = imdecode(result.final_labeling, 'png');
mask = mask - 1;
label_names = result.refined_labels;
pose = result.pose;

%imwrite(label_mask,'output.jpg')
imwrite(mask,'output.png')
save('mask.mat','mask')
save('names.mat','label_names')
save('pose.mat','pose')
%show_parsing(result.image, result.final_labeling, result.refined_labels);
save('output.mat','result')

%profile off
%profile('info')
%profsave(profile('info'),'myprofile_results')
toc
return


%             image: [1x18427 uint8]
%              pose: [1x106 double]
%    refined_labels: {31x1 cell}
%    final_labeling: [1x2231 uint8]
