import streamlit as st
import pandas as pd
import requests
import xml.etree.ElementTree as ET 
from bs4 import BeautifulSoup
from vnstock3 import Vnstock





def get_all_stocks():
    stock = Vnstock().stock(source='VCI')
    df = stock.listing.all_symbols()
    return df



def parse_xml_to_dataframe(xml_text):
    # Parse the XML
    soup = BeautifulSoup(xml_text, "xml")
    
    # Find the relevant CDATA section containing the HTML table
    cdata_section = soup.find("update", id="pt9:t1").text
    
    # Parse the HTML table content
    html_soup = BeautifulSoup(cdata_section, "html.parser")
    
    # Extract table headers
    headers = []
    for header in html_soup.find_all("th", role="columnheader"):
        headers.append(header.get_text(strip=True))
    
    # Extract table rows
    rows = []
    for row in html_soup.find_all("tr", role="row"):
        cells = []
        for cell in row.find_all("td"):
            # Extract text and remove any extra spaces
            cells.append(cell.get_text(strip=True))
        if cells:
            rows.append(cells)
    
    # Create DataFrame
    df = pd.DataFrame(rows, columns=headers)
    return df

import requests
from urllib.parse import urlencode

def get_income_statement(page_number=1):
    url = "https://congbothongtin.ssc.gov.vn/faces/NewsSearch?Adf-Window-Id=v6vo6htkv&Adf-Page-Id=3"
    
    payload_dict = {
        'pt9:it8112': '',
        'pt9:id1': '',
        'pt9:id1::pop::cd::mSel': '5',
        'pt9:id1::pop::cd::ys': '2024',
        'pt9:id2': '',
        'pt9:id2::pop::cd::mSel': '5',
        'pt9:id2::pop::cd::ys': '2024',
        'pt9:it1': '0100100008',
        'pt9:t1:_afrFltrMdlc3': '',
        'pt9:t1:_afrFltrMdlc8': '',
        'pt9:t1:_afrFltrMdlc111': '',
        'pt9:t1:id3': '',
        'pt9:t1:id3::pop::cd::mSel': '5',
        'pt9:t1:id3::pop::cd::ys': '2024',
        'org.apache.myfaces.trinidad.faces.FORM': 'f1',
        'Adf-Window-Id': 'v6vo6htkv',
        'Adf-Page-Id': '3',
        'javax.faces.ViewState': '!-yaly3nihg',
        'oracle.adf.view.rich.DELTAS': '{pt9:t1={viewportSize=16}}',
        'event': 'pt9:b1',
        'event.pt9:b1': '<m xmlns="http://oracle.com/richClient/comm"><k v="type"><s>action</s></k></m>',
        'oracle.adf.view.rich.PROCESS': 'f1,pt9:b1'
    }
    
    headers = {
        'Accept': '*/*',
        'Accept-Language': 'en-US,en;q=0.9,vi;q=0.8,vi-VN;q=0.7,fr-FR;q=0.6,fr;q=0.5,de;q=0.4',
        'Adf-Ads-Page-Id': '7',
        'Adf-Rich-Message': 'true',
        'Connection': 'keep-alive',
        'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
        'Cookie': 'JSESSIONID=Q5sHWvh4cNdGSlkPiIHq1BtY54VNJENWXlFINxrtgYjHJ8kHiOws!1401112399; JSESSIONID=-dcIVjkuXXlb1D3jbfmHA9xVTy1_n2sp9MSS9tFaE8Nmib-HBq0Q!1401112399',
        'DNT': '1',
        'Origin': 'https://congbothongtin.ssc.gov.vn',
        'Referer': 'https://congbothongtin.ssc.gov.vn/faces/NewsSearch',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-origin',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
        'sec-ch-ua': '"Google Chrome";v="125", "Chromium";v="125", "Not.A/Brand";v="24"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"macOS"'
    }
    
    payload = urlencode(payload_dict)
    
    response = requests.post(url, headers=headers, data=payload)
    
    xml = response.text
    
    df = parse_xml_to_dataframe(xml)
    
    return df