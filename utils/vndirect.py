
import requests
import pandas as pd

def get_fund_ratios():
    url = 'https://finfo-api.vndirect.com.vn/v4/fund_ratios?q=reportDate%3Agte%3A2021-12-03~ratioCode%3AIFC_HOLDING_COUNT_CR&size=1000'
    
    headers = {
        'Accept': '*/*',
        'Accept-Language': 'en-US,en;q=0.9,vi;q=0.8,vi-VN;q=0.7,fr-FR;q=0.6,fr;q=0.5,de;q=0.4',
        'Connection': 'keep-alive',
        'Content-Type': 'application/json',
        'DNT': '1',
        'Origin': 'https://dstock.vndirect.com.vn',
        'Pragma': 'no-cache',
        'Referer': 'https://dstock.vndirect.com.vn/',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-site',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
        'sec-ch-ua': '"Google Chrome";v="119", "Chromium";v="119", "Not?A_Brand";v="24"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"macOS"',
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        return f"Error: {response.status_code}"

def load_fund_ratios_to_df(data):
    # """
    # {
    # "data": [
    #     {
    #         "code": "ILB",
    #         "type": "IFC",
    #         "period": "1M",
    #         "ratioCode": "IFC_HOLDING_COUNT_CR",
    #         "reportDate": "2022-09-30",
    #         "value": 1.0
    #     },
    # ]
    # }
    # """
    data = data['data']
    df = pd.DataFrame(data)
    df['reportDate'] = pd.to_datetime(df['reportDate'])
    df = df.sort_values(by=['reportDate'])

    return df
