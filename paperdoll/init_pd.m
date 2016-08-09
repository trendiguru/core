function [mask,label_names,pose] = init_pd(root)
    global pd_config

    tic

    paperdoll_pipeline_path = strcat(root, '/data/paperdoll_pipeline.mat')
    pd_config = load(paperdoll_pipeline_path, 'config');
    % config = evalin('base', 'load(paperdoll_pipeline_path, ''config'');');
    addpath(genpath(root))
    
    pd_config{1}.scale = 200;
    pd_config{1}.model.thresh = -2;

    toc

return
