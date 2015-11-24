%this is a matlab file that takes the unzipped fashionista dataset of 685 pixel-level-annotated images and saves as bitmaps with
%a consistent labelling (shown in 'categories' below)

categories=    {
    'null',
    'tights',
    'shorts',
    'blazer',
    't-shirt',
    'bag',
    'shoes',
    'coat',
    'skirt',
    'purse',
    'boots',
    'blouse',
    'jacket',
    'bra',
    'dress',
    'pants',
    'sweater',
    'shirt',
    'jeans',
    'leggings',
    'scarf',
    'hat',
    'top',
    'cardigan',
    'accessories',
    'vest',
    'sunglasses',
    'belt',
    'socks',
    'glasses',
    'intimate',
    'stockings',
    'necklace',
    'cape',
    'jumper',
    'sweatshirt',
    'suit',
    'bracelet',
    'heels',
    'wedges',
    'ring',
    'flats',
    'tie',
    'romper',
    'sandals',
    'earrings',
    'gloves',
    'sneakers',
    'clogs',
    'watch',
    'pumps',
    'wallet',
    'bodysuit',
    'loafers',
    'hair',
    'skin'}

for k=1:685
    pd_pred=predictions_paperdoll(k);
    for l=1:length(categories)
        if(~ strcmp(pd_pred.labels{l}, categories{l}))
            disp('unexpected labels')
            disp(pd_pred.labels{l})
            disp(a{l})
        end
    end
end

for k=1:685
    pd_pred=predictions_paperdoll(k);
    im=imdecode(pd_pred.labeling,'png');
    imshow(im)
    size(im)
    mystr=sprintf('fashionista_labelmask_%d.bmp',k)
    imwrite(im,mystr)
end