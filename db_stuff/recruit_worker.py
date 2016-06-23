from recruit_API import GET_ByGenreId, process_items
from recruit_constants import recruitID2generalCategory, api_stock
from time import time

def genreDownloader(genreId, loghandler):
    start_time = time()
    success, response_dict = GET_ByGenreId(genreId, limit=100, instock=True)
    if not success:
        print ('GET failed')
        return
    if genreId[1] == '1':
        gender = 'Female'
    else:
        gender = 'Male'
    new_items = total_items = 0
    category = recruitID2generalCategory[genreId]
    sub = [x for x in api_stock if x['genreId'] == genreId][0]['category_name']
    new_inserts, total = process_items(response_dict["itemInfoList"], gender, category)
    new_items += new_inserts
    total_items += total
    pageCount = int(response_dict['pageCount'])
    if pageCount > 999:
        pageCount = 999
    for i in range(2, pageCount + 1):
        success, response_dict = GET_ByGenreId(genreId, page=i, limit=100, instock=True)
        if not success:
            continue
        new_inserts, total = process_items(response_dict["itemInfoList"], gender, category)
        new_items += new_inserts
        total_items += total
    end_time = time()
    summery = 'genreId: %s, Topcategory: %s, Subcategory:%s, total: %d, new: %d, download_time: %d' \
              % (genreId, category, sub, total_items, new_items, (end_time-start_time))
    loghandler.info(summery)
    print(sub + ' Done!')