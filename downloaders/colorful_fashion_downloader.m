load fashon_parsing_data.mat

fashionista_vals=[1,5, 6, 28, 4, 12, 8, 15,57, 55,22,19,20,16,21,7,3,56,9,29,32,27,17]

for f = 1:length(colorful_fashion_parsing_categories)
theval = (f);
newval = fashionista_vals(theval);

s=sprintf(' %d %s %d %s ',theval, colorful_fashion_parsing_categories{theval}{1},newval,f_labels{newval}{1});
disp(s);
end


for f = 1:2682
%for f = 1:3
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
s=sprintf('%d %d %s %d %s ',ind,theval, colorful_fashion_parsing_categories{theval}{1},newval,f_labels{newval}{1});
%disp(s);
out(i,j)=newval;
end
end
c=cast(out,'uint8');
name = strrep(name, '.jpg', '.png')
imwrite(c,name);
end

