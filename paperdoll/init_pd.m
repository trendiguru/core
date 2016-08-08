function [mask,label_names,pose] = init_pd(root)
    tic

    load strcat(root, "/data/paperdoll_pipeline.mat") config;
    addpath(genpath(root))
    
    config{1}.scale = 200;
    config{1}.model.thresh = -2;
    
    toc

return
