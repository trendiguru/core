disp(['starting pd solo, ,let me give you a han'])
%profile on

load data/paperdoll_pipeline.mat config;
addpath(genpath('.'))
input_image = imread('../core/images/male1.jpg');
input_sample = struct('image', imencode(input_image, 'jpg'))
config{1}.scale = 200;
config{1}.model.thresh = -2;

result = feature_calculator.apply(config, input_sample)

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
%profsave(profile('info'),'myprofile_results')

return
