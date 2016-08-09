function [mask,label_names,pose] = pd(image_filename)
    global pd_config
    mask = []; 
    label_names = []; 
    pose = [];
    % config = evalin('base', 'config');

    tic
    
    try
        image_array = imread(image_filename);
    catch     %i think there may be cases where the ml read starts before the python write finishes
        disp('debug3.5 (try catch')
    end

    input_sample = struct('image', image_array);
    result = feature_calculator.apply(pd_config, input_sample);

    if ~ isfield(result, 'final_labeling')
        % paperdoll failed to return result
        disp(['paperdoll failed to get result for ',image_filename])
        return
    end

    mask = imdecode(result.final_labeling, 'png');
    
    mask = mask - 1;
    label_names = result.refined_labels;
    pose = result.pose;
    
    toc

return
