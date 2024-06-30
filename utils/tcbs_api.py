from playwright.sync_api import sync_playwright, Playwright
import requests
import json
import subprocess
import datetime
import random


class TCBSAPI:
    def __init__(self, auth_token):
        self.auth_token = auth_token

    def make_request(self, url, method='GET', data=None):
        """Helper method to make API requests."""
        if self.auth_token is None:
            raise ValueError('Auth token is required to make requests.')
        headers = {
            'authorization': f'Bearer {self.auth_token}',
            'accept': 'application/json',
            'accept-language': 'vi',
            'dnt': '1',
            'origin': 'https://tcinvest.tcbs.com.vn',
            'pragma': 'no-cache',
            'referer': 'https://tcinvest.tcbs.com.vn/',
            'sec-ch-ua': '"Google Chrome";v="125", "Chromium";v="125", "Not.A/Brand";v="24"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"macOS"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-site',
            'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/'
        }

        if method == 'POST':
            response = requests.post(
                url, headers=headers, data=json.dumps(data))
        elif method == 'DELETE':
            response = requests.delete(url, headers=headers)
        else:
            response = requests.get(url, headers=headers)

        response.raise_for_status()
        
        try:
            return response.json()
        except:
            return response.text

    def place_order(self, sub_account_id, account_id, side, symbol, ref_id, price, volume, order_type, pin):
        """Places an order on the TCBS trading platform."""
        url = 'https://apiext.tcbs.com.vn/iftrading/v1/order/place'
        data = {
            'subAccountId': sub_account_id,
            'accountId': account_id,
            'side': side,
            'symbol': symbol,
            'refId': ref_id,
            'price': price,
            'volume': volume,
            'orderType': order_type,
            'pin': pin,
            'cmd': 'Web.newOrder'
        }
        return self.make_request(url, method='POST', data=data)

    def request_otp(self, tcbs_id, iotp_ticket, session, browser_info, duration, type_otp, device_info):
        url = 'https://apiext.tcbs.com.vn/otp/v1/fotp/auth'
        data = {
            'tcbsId': tcbs_id,
            'iotpTicket': iotp_ticket,
            'session': session,
            'browserInfo': browser_info,
            'duration': duration,
            'typeOtp': type_otp,
            'deviceInfo': device_info
        }
        return self.make_request(url, method='POST', data=data)

    def need_otp(self):
        url = 'https://apiext.tcbs.com.vn/otp/v1/dummy/need_otp'
        return self.make_request(url)

    def place_stock_order(self, account_id, side, symbol, price, volume, order_type):
        url = f'https://apiext.tcbs.com.vn/moka/v1/accounts/{account_id}/orders'
        data = {
            'object': 'order',
            'execType': side,
            'symbol': symbol,
            'priceType': order_type,
            'price': price,
            'quantity': volume
        }
        return self.make_request(url, method='POST', data=data)

    def preorder_stock(self, tcbs_id, custodyId, account_id, type, symbol, price, price_type, volume, account_type='NORMAL', start_date=None, end_date=None):
        url = 'https://apiext.tcbs.com.vn/anatta/v1/orders/preorder'

        if start_date is None:
            start_date = datetime.datetime.now().strftime('%Y-%m-%d')

        if end_date is None:
            end_date = (datetime.datetime.now() +
                        datetime.timedelta(days=30)).strftime('%Y-%m-%d')

        if price_type == 'ATO' or price_type == 'ATC':
            price = None
        # random number length 7
        id = ''.join([str(random.randint(0, 9)) for _ in range(7)])
        
        # round price to 100x
        price = int(price) // 100 * 100
        
        data = {
            'id': id,  # 'id': '1555797
            'tcbsId': tcbs_id,
            'code105c': custodyId,
            'accountNo': account_id,
            'accountType': account_type,
            'symbol': symbol,
            'volume': volume,
            'status': '',
            'startDate': start_date,
            'endDate': end_date,
            'execType': type,
            'priceType': price_type,
            'orderPrice': price,
            'matchedVolume': None,
            'isUpdate': False,
            'isCancel': False,
            'flexOrders': None
        }
        print("Preorder data: ", json.dumps(data))
        response = self.make_request(url, method='POST', data=data)

        if response.get('status') == 'error':
            print(response.data)
            raise ValueError(response.get('message'))
        
        print("Preorder response: ", response)
        return response

    def get_balance_info(self, username):
        url = f'https://apiext.tcbs.com.vn/profile-r/v2/get-profile/by-username/{username}?fields=bankSubAccounts,bankAccounts'
        return self.make_request(url)

    def get_hawkeye_balance(self, tcbs_id):
        url = f'https://apiextaws.tcbs.com.vn/hawkeye/v1/{tcbs_id}/balances'
        return self.make_request(url)

    def get_assets_info(self):
        url = 'https://apiext.tcbs.com.vn/asset-hub/v2/asset/overview?reload=true&accountNo='
        return self.make_request(url)

    def get_account_info(self, tcbs_id):
        url = f'https://apiext.tcbs.com.vn/profile/v1/profiles/{tcbs_id}?fields=accountStatus,personalInfo,basicInfo'
        return self.make_request(url)

    
    def get_market_calendar(self, tcbs_id, from_date, to_date, time_frame='1M', src='ALL', role='CUS'):
        if tcbs_id is None:
            raise ValueError('tcbs_id is required to get market calendar.')
        
        url = f'https://apiextaws.tcbs.com.vn/icalendar-service/v1/event/market/{tcbs_id}/list?fDate={from_date}&tDate={to_date}&tFrame={time_frame}&src={src}&role={role}'
        
        return self.make_request(url)
    
    
    def get_pending_orders(self, tcbs_id):
        # https://apiext.tcbs.com.vn/anatta/v1/0001964802/orders?isPending=true
        url = f'https://apiext.tcbs.com.vn/anatta/v1/{tcbs_id}/orders?isPending=true'
        
        return self.make_request(url)
    
    def cancel_preorder(self, order_id):
        # DELETE https://apiext.tcbs.com.vn/anatta/v1/orders/preorder/1581551
        url = f'https://apiext.tcbs.com.vn/anatta/v1/orders/preorder/{order_id}'
        
        return self.make_request(url, method='DELETE')
    
    def get_stock_noti(self, tcbs_id, ticker, lastVisitedId='0', size=50):
        print("Get stock noti: ", tcbs_id, ticker, lastVisitedId, size)
        # https://apiextaws.tcbs.com.vn/moksha/v1/users/0001964802/notification?lastVisitedId=0&size=50&from=&to=&ticker=VND
        url = f'https://apiextaws.tcbs.com.vn/moksha/v1/users/{tcbs_id}/notification?lastVisitedId={lastVisitedId}&size={size}&ticker={ticker}&from=&to='
        
        return self.make_request(url)
    
    def get_stock_noti_all(self, tcbs_id, ticker, lastVisitedId='0', size=50):
        data  = []
        
        while True:
            res = self.get_stock_noti(tcbs_id, ticker, lastVisitedId, size)
            dat = res.get('notifications')
            
            if dat is None or len(dat) == 0:
                break
            
            data += dat
          
            lastVisitedId = res[-1]['id']
            
        return data