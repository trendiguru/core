"""
playground for testing the ebay API
"""
import requests
import json

def GET_gnereid(genreid):
    req = requests.get("http://itemsearch.api.ponparemall.com/1_0_0/search/?"
                       "key=731d157cb0cdd4146397ef279385d833"
                       "&genreId="+str(0100000000) +                        "&format=json"
                       "&limit=100"
                       "&page=1"
                       "&inStockFlg	=0")
    dic = json.dumps(req.text)
    print (dic)

GET_gnereid(0100000000)
# if __name__=='__main__':
#     # db.ebay_global_Female.delete_many({})
#     # db.ebay_global_Male.delete_many({})
#     start = time()
#     top_categories = getTopCategories()
#     stage1 = time()
#     print ('getTopCategories duration = %s' %(str(stage1-start)))
#     brand_info = fill_brand_info(top_categories)
#     stage2 = time()
#     print ('fiil_brand_info duration = %s' % (str(stage2 - stage1)))
#     download(brand_info)
#     stage3 = time()
#     print ('download duration = %s' % (str(stage3 - stage1)))
#
#     print ('done!')
# else:
#     start = time()
#     top_categories = getTopCategories()
#     stage1 = time()
#     print ('getTopCategories duration = %s' % (str(stage1 - start)))
#     brand_info = fill_brand_info(top_categories)
#     stage2 = time()
#     print ('fiil_brand_info duration = %s' % (str(stage2 - stage1)))
#     download(brand_info)
#     stage3 = time()
#     print ('download duration = %s' % (str(stage3 - stage1)))
#     print ('done!')

# for x in brand_info:
#     for y in x['children']:
#         print y['brands']
#         raw_input()

"""
work flow
1. create japanese category list -> get category id
    catagories: number are not accurate
    1. ladies' fashion       => 0100000000  - 547K total - 307K instock
        a. Outer/coat/jacket => 0101000000  -  43K       -  29K
            coats - balmacaan=> 0101010000  to  0101050000
            jacket-          => 0101060000  to  0101110000 and 0101140000
            outer            => 0101120000  to  0101130000


        b. dresses - 14K
        c. one-piece - 69K
        d. suits - 3.5K
        e. skirt - 15K
        f. tops - 114K - 010100000
        g. bottoms - 33K
        h. kimono - 85K
        i. swimsuits - 13K
        j. tunics - 32K


    2. men's fashion - 0200000000 - 355K - 242K in stock
2. query API with genreid + japanese category
    a. get number of items
    b. call pagenumber +1
3. insert 100 items to collection
    for each:
    a. check if img link is valid - ig no link the url ends with 'noimage'
    b. check prior existence by item id and by hash
    c. insert item
    d. call segmentation net - args = mongo id, img_url, collection name, category
    e. segmentation worker calls fp with the mask
        i. if err -> deletes from collection
notice that the images are not so so
-> find  strategy to select best img out of 3
    -> first find face -> than crop-> not good! some imgs dont have faces and are good
    -> run all three through the net
        -> for each calculate certainty in each blob
            -> if certainty surpass 80% for more than one -> choose the biggest blob
-> net might not return the right category
    -> in that case we will want to use near categories
        ->e.g, dress = skirt + top
-> max page number is 999
    -> that means that max items returned 99900
-> relevent return fields

->get tracking id from kyle






"""
