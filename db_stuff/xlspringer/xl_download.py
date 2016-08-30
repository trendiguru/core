from __future__ import print_function
import xmltodict


def process_xml(file_name):
    f = open(file_name)
    d = xmltodict.parse(f)
    d1 = d['Products']
    d2 = d1['Product']
    total_items = len(d2)
    one_percent = int(total_items/100)
    print ('[', end='')
    tmp = 0
    cats = []
    for x, item in enumerate(d2):
        if divmod(x, one_percent)[0] > tmp:
            print ('#', end='')
            tmp += 1

        item_id = item['@ArticleNumber']
        #todo if...

        category_id = item['CategoryPath']['ProductCategoryID']

        price = {'price': item['Price']['Price'],
                 'currency': item['Price']['CurrencySymbol'],
                 'priceLabel': item['Price']['DisplayPrice']}

        click_url = item['Deeplinks']['Product']

        image_url = item['Images']['Img'][0]['URL']

        short_d = item['Details']['DescriptionShort']
        long_d = item['Details']['Description']
        brand = item['Details']['Brand']
        title = item['Details']['Title']

        new_item = {'id': item_id,
                    'category_id': category_id,
                    'clickUrl': click_url,
                    'images': {'XLarge': image_url},
                    'price': price,
                    'shortDescription': short_d,
                    'longDescription': long_d,
                    'brand': brand}
                    # 'categories': category,
                    # 'sub_category': sub_category,
                    # 'raw_info': attributes,
                    # 'tree': family_tree,
                    # 'status': {'instock': True, 'days_out': 0},
                    # 'fingerprint': {},
                    # 'gender': gender,
                    # 'img_hash': img_hash,
                    # 'p_hash': p_hash,
                    # 'download_data': {'dl_version': today_date,
                    #                   'first_dl': today_date,
                    #                   'fp_version': fingerprint_version}


        if category_id not in cats:
             cats.append(category_id)

    for c in cats:
        print (c)

if __name__ == "__main__":
    filename = '/home/developer/yonti/affilinet_products_5420_777756.xml'
    process_xml(filename)