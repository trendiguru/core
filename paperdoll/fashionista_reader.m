%this is a matlab file that takes the unzipped fashionista dataset of 685 pixel-level-annotated images and saves as bitmaps with
%a consistent labelling (shown in 'categories' below)

pd_categories=    {
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

%to translate these to shopstyle categories you can use constants.paperdoll_shopstyle_women
%where the key is pd label and value is shopstyle equivalent
%reproduced below for your reading pleasure
%paperdoll_shopstyle_women = {'top': 'womens-tops', 'pants': 'womens-pants', 'shorts': 'shorts', 'jeans': 'jeans',
%                             'jacket': 'jackets', 'blazer': 'blazers', 'shirt': 'womens-tops', 'skirt': 'skirts',
%                             'blouse': 'womens-tops', 'dress': 'dresses', 'sweater': 'sweaters',
%                             't-shirt': 'tees-and-tshirts', 'cardigan': 'cardigan-sweaters', 'coat': 'coats',
%                             'suit': 'womens-suits', 'vest': 'vests', 'sweatshirt': 'sweatshirts',
%                             'jumper': 'v-neck-sweaters', 'bodysuit': 'shapewear', 'leggings': 'leggings',
%                             'stockings': 'hosiery', 'tights': 'leggings'}

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