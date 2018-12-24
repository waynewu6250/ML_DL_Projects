import urllib
import requests
from bs4 import BeautifulSoup
import os
import re

save_path = 'files/mayday.txt'
chou = 'https://mojim.com/twh100951.htm'
eason = 'https://mojim.com/twh100111-1.htm'
jj = 'http://mojim.com/twh102520.htm'
mayday = 'https://mojim.com/twh100012.htm'

if os.path.exists('files') is False:
    os.mkdir('files')

url = mayday

fhand = urllib.request.urlopen(url).read()
soup = BeautifulSoup(fhand, 'html.parser')

for (i,html) in enumerate(soup.find_all('a',{'href': re.compile(r'^/twy.*')})):
    try:
        # Get soup
        new_url = 'https://mojim.com'+html['href']
        fhand2 = urllib.request.urlopen(new_url)
        soup2 = BeautifulSoup(fhand2, 'html.parser')
        # Get text
        text = soup2.find_all('dd')
        li = [x for x in str(text[6].get_text).split('<br/>') if x != '']
        li = [x for x in li if re.search(r'^(\[)|(\<)|作|(^編)',x) == None]
        if li == []:
            continue
        with open(save_path, 'a') as f:
            for l in li:
                f.write(l+'\n')
            f.write('end\n')
        
        print("Complete Song:%d"%i)
    except:
        continue
    
