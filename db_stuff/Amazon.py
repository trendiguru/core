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

sample request:
http://webservices.amazon.com/onca/xml?
Service=AWSECommerceService&
AWSAccessKeyId=[AWS Access Key ID]&
AssociateTag=[Associate ID]&
Operation=ItemSearch&
Brand=Lacoste&
Availability=Available&
SearchIndex=FashionWomen&
Keywords=shirts
&Timestamp=[YYYY-MM-DDThh:mm:ssZ]
&Signature=[Request Signature]
'''