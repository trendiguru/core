function get_pose_boxes = get_boxes(im)
    load('PARSE_model');
    max_side = max(size(im, 1), size(im, 2));
    if max_side > 400
        im = imresize(im, 400/max_side);
    end

    % call detect function
    boxes = detect_fast(im, model, min(model.thresh,-1));
    boxes = nms(boxes, .1); % nonmaximal suppression
    % choosing the right figure, assuming it's the biggest
    chosen = 0; i_chosen = 1;
    % i = number of figures
    for i = 1:size(boxes, 1) % for every figure found
        if boxes(i,104)-boxes(i,2) > chosen
            chosen = boxes(i,104)-boxes(i,2);
            i_chosen = i;
        end
    end
    get_pose_boxes = boxes(i_chosen, :);
end