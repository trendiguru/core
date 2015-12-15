function ests = analyze_image(path_to_image)
startup;
clear mex;
global GLOBAL_OVERRIDER;
GLOBAL_OVERRIDER = @lsp_conf;
conf = global_conf();
cachedir = conf.cachedir;
pa = conf.pa;
p_no = length(pa);
note = [conf.note];
diary([cachedir note '_log_' datestr(now,'mm-dd-yy') '.txt']);

% read data
[pos_train, pos_val, pos_test, neg_train, neg_val, tsize] = LSP_data();

% train dcnn
caffe_solver_file = 'external/my_models/lsp/lsp_solver.prototxt';
train_dcnn(pos_train, pos_val, neg_train, tsize, caffe_solver_file);

% train graphical model
model = train_model(note, pos_val, neg_val, tsize);

% testing

%function boxes = test_model(note,model,test)
% boxes = testmodel(name,model,test,suffix)
% Returns candidate bounding boxes after non-maximum suppression

conf = global_conf();

cachedir = conf.cachedir;
par.impyra_fun = conf.impyra_fun;
par.useGpu = conf.useGpu;
%par.useGpu = 0;
par.device_id = conf.device_id;
par.at_least_one = conf.at_least_one;
par.test_with_detection = conf.test_with_detection;
if par.test_with_detection
  par.constrainted_pids = conf.constrainted_pids;
end

num = 1
  all_pos = struct('im', cell(num, 1), 'joints', cell(num, 1), ...
    'r_degree', cell(num, 1), 'isflip', cell(num,1));
  %  fr = frs_pos(ii);
    all_pos(1).im = path_to_image;
  %  all_pos(1).joints = lsp_joints(1:2,joint_order,fr)';
    all_pos(1).r_degree = 0;
    all_pos(1).isflip = 0;

%jr insert
  boxes = cell(1,1);
  %     parfor i = 1:length(test)
  for i = 1:1
%    fprintf([note ': testing: %d/%d\n'],i,length(test));
   disp('detecting fast....')
    box = detect_fast(all_pos,model,model.thresh,par);
        disp(box)
    boxes{1} = nms_pose(box,0.3);
  end
%end jr insert

  if ~isempty(boxes{1})
    % only keep the highest scoring estimation for evaluation.
    boxes{1} = boxes{1}(1,:);
  end
  % visualization
  if 1
    im = imreadx(test(i));
    if ~isempty(boxes{i})
      showskeletons(im, boxes{i}(1,:), conf.pa);
      pause;
    end
  end

ests = conf.box2det(boxes, p_no);
disp(ests)
