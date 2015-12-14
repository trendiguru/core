startup;
clear mex;
global GLOBAL_OVERRIDER;
GLOBAL_OVERRIDER = @lsp_conf;
conf = global_conf();
cachedir = conf.cachedir;
pa = conf.pa;
p_no = length(pa);
note = [conf.note];
diary([cachedir note '_log_' datestr(now,'mm-dd-yy') '.jrout.txt']);

% -------------------------------------------------------------------------
% read data
% -------------------------------------------------------------------------
[pos_train, pos_val, pos_test, neg_train, neg_val, tsize] = LSP_data();
% -------------------------------------------------------------------------
% train dcnn
% -------------------------------------------------------------------------
caffe_solver_file = 'external/my_models/lsp/lsp_solver.prototxt';
train_dcnn(pos_train, pos_val, neg_train, tsize, caffe_solver_file);
% -------------------------------------------------------------------------
% train graphical model
% -------------------------------------------------------------------------
model = train_model(note, pos_val, neg_val, tsize);

%try one image
%boxes = test_model([note,'_LSP'], model, pos_test);
%[pos_train, pos_val, pos_test, neg_train, neg_val, tsize] = LSP_data();

%function boxes = test_model(note,model,test)

%from LSP_data:
par.impyra_fun = conf.impyra_fun;
par.useGpu = conf.useGpu;
par.device_id = conf.device_id;
par.at_least_one = conf.at_least_one;
par.test_with_detection = conf.test_with_detection;
if par.test_with_detection
  par.constrainted_pids = conf.constrainted_pids;
end

%all_pos = struct('im', cell(num, 1), 'joints', cell(num, 1), ...
%    'r_degree', cell(num, 1), 'isflip', cell(num,1));
%  for ii = 1:numel(frs_pos)
%    fr = frs_pos(ii);
%    all_pos(ii).im = sprintf(lsp_imgs,fr);
%    all_pos(ii).joints = lsp_joints(1:2,joint_order,fr)';
%    all_pos(ii).r_degree = 0;
%    all_pos(ii).isflip = 0;
%  end
num = 1
my_test = struct('im', cell(num, 1), 'joints', cell(num, 1), ...
    'r_degree', cell(num, 1), 'isflip', cell(num,1));
%fr = frs_pos(ii);
%my_test.im = sprintf(lsp_imgs,fr);
my_test(1).im = 'dataset/LSP/images/im0001.jpg';
%my_test.joints = lsp_joints(1:2,joint_order,fr)';
my_test(1).r_degree = 0;
my_test(1).isflip = 0;

box = detect_fast(my_test,model,model.thresh,par);
 % was test(i) instead of mytest
disp(box)

% -------------------------------------------------------------------------
% testing
% -------------------------------------------------------------------------

%boxes = test_model([note,'_LSP'], model, pos_test);

% -------------------------------------------------------------------------
% evaluation
% -------------------------------------------------------------------------

%eval_method = {'strict_pcp', 'pdj'};
%fprintf('============= On test =============\n');
%ests = conf.box2det(boxes, p_no);
% generate part stick from joints locations
%for ii = 1:numel(ests)
%  ests(ii).sticks = conf.joint2stick(ests(ii).joints);
%  pos_test(ii).sticks = conf.joint2stick(pos_test(ii).joints);
%end
%show_eval(pos_test, ests, conf, eval_method);

diary off;
clear mex