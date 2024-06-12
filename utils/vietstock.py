import pandas as pd
import requests as native_requests
import requests_cache
from urllib.parse import urlencode
import datetime as dt

from bs4 import BeautifulSoup
from utils.misc import retry

CACHE_TTL = 60 * 60 * 24  # 1 day
requests = requests_cache.CachedSession('cache/vietstock', expire_after=CACHE_TTL, allowable_codes=[200])

def parse_news_response(html_content):
    html_content = f"<html>{html_content}</html>"

    # Parse the HTML content
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find the paging information
    paging_info = soup.find('div', class_='m-b pull-left').text.strip()
    total_pages = paging_info.split('/')[-1].strip()

    # Find the current page
    current_page = soup.find('li', class_='active').text.strip()

    # Extract all page numbers from pagination
    page_numbers = [li.text.strip() for li in soup.select('ul.pagination li a') if li.text.strip().isdigit()]

    # Extract previous and next page information
    # previous_page = soup.find('li', id='page-previous ').a['page']
    next_page_el = soup.find('li', id='new-page-next ')
    
    next_page = next_page_el.a['page'] if next_page_el else None

    # Extract first and last page information
    # first_page = soup.find('li', id='page-first ').a['page']
    # last_page = soup.find('li', id='page-last ').a['page']

    table = html_content
    
    return {
        'total_pages': total_pages,
        'current_page': current_page,
        'page_numbers': page_numbers,
        # 'previous_page': previous_page,
        'next_page': next_page,
        # 'first_page': first_page,
        # 'last_page': last_page,
        'table': table
    }

@retry(times=5, exceptions=(native_requests.exceptions.RequestException,), delay=10)
def get_stock_news(code, page=1, page_size=20, from_date=None, to_date=None):

    url = "https://finance.vietstock.vn/View/PagingNewsContent"
    
    if from_date is None:
        from_date = (dt.datetime.now() - dt.timedelta(days=365))
        
    if to_date is None:
        to_date = dt.datetime.now()
        
    if isinstance(from_date, dt.datetime):
        from_date = from_date.strftime('%d/%m/%Y')
        
    if isinstance(to_date, dt.datetime):
        to_date = to_date.strftime('%d/%m/%Y')

    payload_dict = {
        'view': '1',
        'code': code,
        'type': '1',
        'fromDate': from_date,
        'toDate': to_date,
        'channelID': '-1',
        'page': str(page),
        'pageSize': str(page_size),
    }
    payload = urlencode(payload_dict)

    headers = {
    'Accept': 'text/html, */*; q=0.01',
    'Accept-Language': 'en-US,en;q=0.9,vi;q=0.8,vi-VN;q=0.7,fr-FR;q=0.6,fr;q=0.5,de;q=0.4',
    'Connection': 'keep-alive',
    'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
    'Cookie': '__gads=ID=23a84ba3642985dd-225dd1c1efe40031:T=1697534635:RT=1697534635:S=ALNI_MYrkFZ27_DnnSrsceg2yfjNe1N4NQ; __gpi=UID=00000c651a2129ed:T=1697534635:RT=1697534635:S=ALNI_MYGsWgiOZjGVAvDSSFeqB5SiTe_hw; _cc_id=b79ffaf0fbfc19b4fcaa985468c04c5a; dable_uid=undefined; cto_bidid=ASnYGV8xMkJiQlNPNGFCTVJVcWVCJTJCQ3FzU0VuZzFBMVJjSW9FJTJGN291dUdqdzF5ZUxKUUVvb1Zlek0xUlhuOTRPc3RxWE9xQUtKOHRYN0YlMkJBMWZSYmplN3N5MkozbUw5dlAyUTIlMkZqdDNWMHp1OHJJJTNE; cto_bundle=knrmOF80eDNUQm4wQk1lekg0VGFGVUwwb0V6JTJCZnklMkYxTkFKdGNpWmJwZElWcDZORVo4YTBaUGNLalhpM2pzR3BtTnpHdjB6cEFJeVQxVDlMMkNWaldMSVFhWW44TWxVOWZxU3lsZU54UjRYbWlzcEUlMkIxeHg1NVNydXFoQTUlMkZRVUk5T3BUUFB5RHZKaUNGbEJXdmIlMkIyWHoyc1kzVE1oOWxYSXJQT1hseTZaSSUyRlcxRmslM0Q; _ga=GA1.2.1939511784.1697534635; _ga_EXMM0DKVEX=GS1.1.1709646248.11.0.1709646248.60.0.0; language=vi-VN; Theme=Light; AnonymousNotification=; ASP.NET_SessionId=233fm0kc4cc1paa2mka1xzgw; __RequestVerificationToken=ZQsefvxPtvI1C4ytred5kFaQU3oHmPJNdLSGKx3fY_Il6RX2spYc3X54eeqwE0N9Kf7aCdWeMQlndZ1vevvDg5ze9NXdmnuKSN1ZjaN-_Ks1; finance_viewedstock=DIC,CONGTHANH,VHM,CAP,',
    'DNT': '1',
    'Origin': 'https://finance.vietstock.vn',
    'Referer': 'https://finance.vietstock.vn/CAP/tin-moi-nhat.htm',
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'same-origin',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
    'X-Requested-With': 'XMLHttpRequest',
    'sec-ch-ua': '"Google Chrome";v="125", "Chromium";v="125", "Not.A/Brand";v="24"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"macOS"'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    data = parse_news_response(response.text)
    
    return data

def load_stock_news_to_df(html: str):
    dfs = pd.read_html(html)
    
    df = dfs[0]
    
    df.columns = ['date', 'title']
    
    # parse the date 12/07/2023
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
    
    # set the index to date
    df.set_index('date', inplace=True)
    
    return df

def get_stock_news_all(code, page=1, page_size=20, from_date=None, to_date=None):
    results_df = pd.DataFrame()
    
    data = get_stock_news(code, page, page_size, from_date, to_date)
    data_df = load_stock_news_to_df(data['table'])
    
    results_df = pd.concat([results_df, data_df])
    
    next_page = data['next_page']
        
    while next_page is not None:
        data = get_stock_news(code, next_page, page_size, from_date, to_date)
        next_page = data['next_page']
        # return data
        data_df = load_stock_news_to_df(data['table'])
        results_df = pd.concat([results_df, data_df])
    
    return results_df