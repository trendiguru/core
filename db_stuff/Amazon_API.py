'''
useful request parameters:
1. Availability = available
2. Brand
3. BrowseNode - > category id
4. ItemPage
5. Keywords
6. MaximumPrice - 32.42 -> 3242
7. MinimumPrice - same
8. SearchIndex  = localrefernce name


parameters = {
    'AWSAccessKeyId': 'AKIAIQJZVKJKJUUC4ETA',
    'AssociateTag': 'fazz0b-20',
    'Availability': 'Available',
    'Brand': 'Lacoste',
    'Keywords': 'shirts',
    'Operation': 'ItemSearch',
    'SearchIndex': 'FashionWomen',
    'Service': 'AWSECommerceService',
    'Timestamp': time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    'ResponseGroup': 'ItemAttributes,Offers,Images,Reviews'}


AWSAccessKeyId = AKIAIQJZVKJKJUUC4ETA
AWSAccessKeyPwd = r82svvj4F8h6haYZd3jU+3HkChrW3j8RGcW7WXRK

IMPORTANT!!!
ItemPage can only get numbers from 1 to 10
but each page return 10 results at max
for example, under women's shirts there are 100,000 pages for 1 million items
therefore, we need to divide the requests in a way that less then 100 results will return per search

hierarchy:
7141123011 -> Clothing, Shoes & Jewelry
    7141124011 -> Departments
        7147440011-> Women
            1040660     -> Clothing
                1045024     -> dresses
                    2346727011 -> Casual
                2368343011  -> Tops & Tees
            679337011   -> Shoes

'''

