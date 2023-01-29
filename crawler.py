# 导入 beautiful soup4 包，用于抓取网页信息
import re
import time

import bs4 as bs
# 导入 pickle 用于序列化对象
import pickle
# 导入 request 用于获取网站上的源码
import requests
import datetime as dt
import os
from lxml import etree
import pandas_datareader.data as web


class Download_HistoryStock(object):
    def __init__(self, code):
        self.code = code
        self.start_url = "http://quotes.money.163.com/trade/lsjysj_" + self.code + ".html"
        self.headers = {
            "User-Agent": ":Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36"
        }

    def parse_url(self):
        response = requests.get(self.start_url)
        if response.status_code == 200:
            return etree.HTML(response.content)
        return False

    def get_date(self, response):
        # 得到开始和结束的日期
        start_date = ''.join(response.xpath('//input[@name="date_start_type"]/@value')[0].split('-'))
        end_date = ''.join(response.xpath('//input[@name="date_end_type"]/@value')[0].split('-'))
        return start_date, end_date

    def download(self, start_date, end_date):
        download_url = "http://quotes.money.163.com/service/chddata.html?code=0" + self.code + "&start=" + start_date + "&end=" + end_date + "&fields=TCLOSE;HIGH;LOW;TOPEN;LCLOSE;CHG;PCHG;TURNOVER;VOTURNOVER;VATURNOVER;TCAP;MCAP"
        data = requests.get(download_url)
        f = open(self.code + '.csv', 'wb')

        for chunk in data.iter_content(chunk_size=10000):
            if chunk:
                f.write(chunk)

    def run(self):
        html = self.parse_url()
        start_date, end_date = self.get_date(html)
        print(start_date)
        self.download(start_date, end_date)


class StockCode(object):
    def __init__(self):
        self.start_url = "http://quote.eastmoney.com/stocklist.html#sh"
        self.headers = {
            "User-Agent": ":Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36"
        }

    def parse_url(self):
        # 发起请求，获取响应
        response = requests.get(self.start_url, headers=self.headers)
        if response.status_code == 200:
            return etree.HTML(response.content)

    def get_code_list(self, response):
        # 得到股票代码的列表
        node_list = response.xpath('//a/text()')
        print(node_list)
        code_list = []
        for node in node_list:
            try:
                code = re.match(r'.*?\((\d+)\)', etree.tostring(node).decode()).group(1)
                code_list.append(code)
            except:
                continue
        return code_list

    def run(self):
        html = self.parse_url()
        return self.get_code_list(html)


def getDataFromYahoo(reloadSS50=False):
    with open('datasets/SS50tickers.pickle', 'rb') as f:
        tickers = pickle.load(f)
    start = dt.datetime(2010, 1, 1)
    end = dt.datetime.now()
    for ticker in tickers:
        print(str(ticker).split('.')[0])
        his = Download_HistoryStock(str(ticker).split('.')[0])
        his.run()


getDataFromYahoo(True)
