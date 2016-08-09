__author__ = 'stan'
from bs4 import BeautifulSoup
import requests
import re
import time

URL_INPUT = "http://www.chictopia.com/photo/show/1153934-the+calm+before+the+snow-black-ankle-boots-freda-salvador-boots-black-skinny-jeans-dl1961-jeans"

class ExtractRoster(object):              # scraping random page of pages where input is integer in range (1:)

   def __init__(self, npage):
       self.url = "http://www.chictopia.com/browse/people/" + str(npage) + "?g=1"
       self.links = self.get_item_url()

   def get_item_url(self):                # getting list of page urls (9 pages for each page of pages)
       links_list = []
       html_page = requests.get(self.url)
       soup = BeautifulSoup(html_page.content)
       data = soup.find_all("div", {"class": "bold px12 white lh12 ellipsis"})
       for item in data:
           links_list.append("http://www.chictopia.com" + str(item).split('"')[3])
       return links_list


class ExtractData(object):                 # scraping page from page of pages

   def __init__(self, page_url):
       items = []
       try:
           html_page = requests.get(page_url)              # requesting data from html
           self.soup = BeautifulSoup(html_page.content)    # getting soup from the requested data
           self.main_pic = self.get_pic()
           self.add_pics = self.get_add_pics()
           self.tags = self.get_tags()
           self.soup_items = self.get_items()                   # items soup

           for item in self.soup_items:                         # looping through each item in self.soup_items
               items.append(self.get_item_info(item))
           self.items = items

           self.wrap = self.wrap()
       except Exception as exception:
           print exception

   def get_pic(self):                                       # getting main picture

       image = self.soup.find_all("img", {"itemprop": "image"})
       picture_url = str(image[0]).split('"')[11]
       return picture_url

   def get_add_pics(self):                     # getting additional pictures if there are any

       data = self.soup.find_all("div", {"style": "display:inline-block"})

       main_pic_number = re.findall(r'\d+', self.main_pic)[1]
       add_pics = re.findall('src="(.+?)"', str(data))

       add_pictures = []

       for add in add_pics[1:]:
           add_pic_number = (re.findall(r'\d+', add))[1]
           replace = re.sub(main_pic_number, add_pic_number, self.main_pic)
           add_pictures.append(replace)

       if len(add_pictures) > 0:
           return add_pictures
       else:
           return None

   def get_tags(self):                                         # getting list of tags

       tag_list = []
       tags = self.soup.find_all("div", {"class": "left clear px10"})
       for item in tags:
           string = str(item)
           n = re.compile('>(.*?)<', re.DOTALL).findall(string)
           for thing in n:
               if len(thing) > 2:
                   tag_list.append(thing)
       return tag_list

   def get_items(self):                                        # get html soup of clothes items

       keywords_soup = self.soup.find_all("div", {"class": "garmentLinks left"})
       print "\nItems found:", len(keywords_soup), "\n"
       return keywords_soup

   def get_item_info(self, item):                            # getting item information from soup

       item_info = {}
       reg_expr = re.compile('>(.*?)<', re.DOTALL).findall(str(item))
       description = filter(lambda word: len(word)>2, reg_expr)
       item_info["category"] = (description[-1].split(" "))[-1]
       item_info["description"] = description

       return item_info

   def wrap(self):                                        # writing all info about image into dictionary
       pic_info = {}
       pic_info['image-url'] = self.main_pic
       pic_info['tags'] = self.tags
       pic_info['items'] = self.items
       pic_info['other-images'] = self.add_pics

       return pic_info


start = time.time()
page = ExtractData(URL_INPUT)
print "Picture information:", page.wrap
stop = time.time()
print "\nExecution time is", stop - start