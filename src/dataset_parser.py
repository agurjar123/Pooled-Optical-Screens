import requests
from bs4 import BeautifulSoup

base = "https://ftp.ebi.ac.uk/pub/databases/IDR/idr0162-kudo-perturbview/"

def list_dir(url, depth=0):
    if depth > 3:
        return
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    for link in soup.find_all('a'):
        href = link.get('href')
        if href and not href.startswith('?') and href != '../':
            print("  " * depth + href)
            if href.endswith('/'):
                list_dir(url + href, depth + 1)

list_dir(base)