%pose
%{...
%    'right_ankle',...
%    'right_knee',...
%    'right_hip',...
%    'left_hip',...
%    'left_knee',...
%   'left_ankle',...
%    'right_hand',...
%   'right_elbow',...
%    'right_shoulder',...
%    'left_shoulder',...
%    'left_elbow',...
%    'left_hand',...
%    'neck',...
%   'head'...
%}

load fashionista_v0.2.1.mat

for i = 1:685
segmentation = imdecode(truths(i).annotation.superpixel_map, 'png');
clothing_annotation = truths(i).annotation.superpixel_labels(segmentation);
c=cast(clothing_annotation,'uint8');
name=strcat(int2str(i),'_mask.png');
imwrite(c,name);

name=strcat(int2str(i),'_superpixels.tiff')
imwrite(segmentation,name);

pose = truths(i).pose.point
name=strcat(int2str(i),'_pose.txt')
fileID = fopen(name,'w');
fprintf(fileID,'%d %d\n',pose(1,:));
fprintf(fileID,'%d %d\n',pose(2,:));
fprintf(fileID,'%d %d\n',pose(3,:));
fprintf(fileID,'%d %d\n',pose(4,:));
fprintf(fileID,'%d %d\n',pose(5,:));
fprintf(fileID,'%d %d\n',pose(6,:));
fprintf(fileID,'%d %d\n',pose(7,:));
fprintf(fileID,'%d %d\n',pose(8,:));
fprintf(fileID,'%d %d\n',pose(9,:));
fprintf(fileID,'%d %d\n',pose(10,:));
fprintf(fileID,'%d %d\n',pose(11,:));
fprintf(fileID,'%d %d\n',pose(12,:));
fprintf(fileID,'%d %d\n',pose(13,:));
fprintf(fileID,'%d %d\n',pose(14,:));
fclose(fileID);

im = imdecode(truths(i).image, 'jpg');
name=strcat(int2str(i),'_photo.jpg');
imwrite(im,name);

end
