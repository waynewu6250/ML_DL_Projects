import urllib
import requests
from bs4 import BeautifulSoup
import os

def download(url, filename):
    if os.path.exists(filename):
        print('file exists!')
        return
    try:
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    f.flush()
        return filename
    
    except KeyboardInterrupt:
        if os.path.exists(filename):
            os.remove(filename)
        raise KeyboardInterrupt

if os.path.exists('imgs') is False:
    os.mkdir('imgs')

for i in range(1,202):
    #url = 'https://www.zerochan.net/ONE+PIECE?p={}'.format(i)
    #url = 'https://www.imdb.com/title/tt6342474/mediaindex?page={}&ref_=ttmi_mi_sm'.format(i)
    url = 'https://www.desktopnexus.com/tag/naruto/{}'.format(i)

    fhand = urllib.request.urlopen(url).read()
    soup = BeautifulSoup(fhand, 'html.parser')

    for ii, img in enumerate(soup.find_all('img')):
        if img["src"].find('thumbnail') != -1:
            filename = os.path.join('imgs2/'+'page{}-{}.jpg'.format(i,ii))
            download('https:'+img["src"], filename)
    print("Finish scraping page %d"%i)

        
