function [mask,label_names,pose] = pd(image_array)

    mask = [] ;
    label_names = [] ;
    pose = [] ;
    %profile on
    tic
    %disp('debug0')
    load /home/pd_user/paperdoll/data/paperdoll_pipeline.mat config;
    disp('debug1')
    addpath(genpath('.'))
    disp('debug2')


    %read image
    try
        input_image = image_array;
    catch     %i think there may be cases where the ml read starts before the python write finishes
        disp('debug3.5 (try catch')
    end

    input_sample = struct('image', image_array);
    
    config{1}.scale = 200;
    config{1}.model.thresh = -2;

    result = feature_calculator.apply(config, input_sample)

    if ~ isfield(result, 'final_labeling')
        % paperdoll failed to return result
        disp('paperdoll failed to get result for image')
        return
    end

    mask = imdecode(result.final_labeling, 'png');
    
    mask = mask - 1;
    label_names = result.refined_labels;
    pose = result.pose;
    
    toc

return
