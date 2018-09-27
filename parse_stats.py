import json
import simplejson
import requests
import time
import re
import footballdata


url = 'http://www.whoscored.com/Matches/829726/Live/England-Premier-League-2014-2015-Stoke-Manchester-United'
params = {}

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36',
           'X-Requested-With': 'XMLHttpRequest',
           'Host': 'www.whoscored.com',
           'Referer': 'http://www.whoscored.com/'}


response = requests.get(url, params=params, headers=headers)

regex = re.compile("var matchCentreData = (\{.+\});\r\n        var matchCentreEventTypeJson", re.S)
match = re.search(regex, response.text)
# now match.groups(1)[0] will contain the match centre data json blob
match_centre_data = json.loads(match.groups(1)[0])
print(match_centre_data['playerIdNameDictionary']['34693'])


import re

url = 'http://www.whoscored.com/Matches/829726/Live/England-Premier-League-2014-2015-Stoke-Manchester-United'

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36',
           'X-Requested-With': 'XMLHttpRequest',
           'Host': 'www.whoscored.com',
           'Referer': 'http://www.whoscored.com/'}


r = requests.get(url,  headers=headers)

json_req = json.loads(response.text)

from bs4 import BeautifulSoup
soup = BeautifulSoup(r.content, "html.parser")
data_cen = re.compile('var matchCentreData = ({.*?})')
event_type = re.compile('var matchCentreEventTypeJson = ({.*?})')

data = soup.find("a", href="/ContactUs").find_next("script").text
d = json.dumps(data_cen.search(data).group(1))
e = json.dumps(event_type.search(data).group(1))

data_dict = json.loads(d)
event_dict = json.loads(e)
print(event_dict)
print(data_dict)