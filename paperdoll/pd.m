function [mask,label_names,pose] = pd(image_filename)
%todo - check if path already ok,
% check if data already loaded
disp(['the image sent to pd in matlab is:' image_filename])
%todo - check if we cant load this once only (when engine is created)

logname = strcat(image_filename,'.log')
fid = fopen(logname, 'a+');
s = sprintf('starting pd.m analysis of %s\n',image_filename)
fprintf(fid, s);
fclose(fid);

mask = [] ;
label_names = [] ;
pose = [] ;
%profile on
tic
disp('debug0')
load /home/pd_user/paperdoll/data/paperdoll_pipeline.mat config;
disp('debug1')
addpath(genpath('.'))
disp('debug2')


%read image
try
input_image = imread(image_filename);
disp('debug3')
catch     %i think there may be cases where the ml read starts before the python write finishes
    disp('debug3.5 (try catch')
    pause(0.2)
    try
    input_image = imread(image_filename);
    disp('debug3.7 (inner try catch)')
    catch
        warning(['Problem doing imread of ',image_filename]) ;
        fid = fopen('pd_ml_errlog.log', 'a+');
        s = sprintf('problem reading image %s\n',image_filename)
        fprintf(fid, s);
        fclose(fid);
        return
     end
end



input_sample = struct('image', imencode(input_image, 'jpg'));
disp('debug4')
config{1}.scale = 200;
disp('debug5')
config{1}.model.thresh = -2;
disp('debug6')

result = feature_calculator.apply(config, input_sample)
disp('debug7')
%result = feature_calculator.apply(config, input_sample)

if ~ isfield(result, 'final_labeling')
    % paperdoll failed to return result
    disp(['paperdoll failed to get result for ',image_filename])
    fid = fopen('pd_ml_errlog.log', 'a+');
    s = sprintf('result from pd didnt have final labelling for image %s\n',image_filename)
    fprintf(fid, s);
    fclose(fid);
    failname = strcat('/home/jeremy/pd_output/fail_labelling.',image_filename)
    imwrite(input_image,failname)
    return
end


mask = imdecode(result.final_labeling, 'png');
disp('debug8')
mask = mask - 1;
disp('debug9')
label_names = result.refined_labels;
disp('debug10')
pose = result.pose;
disp('debug11')

%imwrite(label_mask,'output.jpg')
imwrite(mask,'output.png')
disp('debug12')
save('mask.mat','mask')
disp('debug13')
save('names.mat','label_names')
disp('debug14')
save('pose.mat','pose')
disp('debug15')
%show_parsing(result.image, result.final_labeling, result.refined_labels);
save('output.mat','result')
disp('debug16')

%profile off
%profile('info')
%profsave(profile('info'),'myprofile_results')
toc
disp('debug17')

logname = strcat(image_filename,'.log')
fid = fopen(logname, 'a+');
s = sprintf('finished pd.m analysis of %s\n',image_filename)
fprintf(fid, s);
fclose(fid);


return


%             image: [1x18427 uint8]
%              pose: [1x106 double]
%    refined_labels: {31x1 cell}
%    final_labeling: [1x2231 uint8]
