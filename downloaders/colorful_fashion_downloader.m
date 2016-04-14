load fashon_parsing_data.mat

fashionista_vals=[0,4, 5, 27, 3, 11, 7, 14,56, 54,21,18,19,15,20,6,2,55,8,28,31,26,16]

%for f = 1:2682
for f = 1:3
a = fashion_dataset(f)
b=a{1}
name = b.img_name
segmentation = b.segmentation;
%clothing_annotation = b.category_label(segmentation);
for i = 1:600
for j=1:400
ind=segmentation(i,j)+1;
theval = b.category_label(ind);
newval = fashionista_vals(theval);
s=sprintf('%d %d %d',ind,theval,newval);
disp(s);
out(i,j)=theval;
end
end
c=cast(out,'uint8');
name = strrep(name, '.jpg', '.png')
imwrite(c,name);
end

