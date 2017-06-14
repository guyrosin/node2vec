from concurrent.futures import ThreadPoolExecutor
import requests
from bs4 import BeautifulSoup
import logging
import json
from datetime import datetime
import calendar
import ftfy

test_mode = False


class WikiEventsCrawler:
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.116 Safari/537.36'
    }

    def __init__(self):
        template_url = 'https://en.wikipedia.org/wiki/Category:{}_{}_events'
        start_year = 1900
        end_year = 2017

        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.WARNING
                            # , filename='crawler_log_%s.log', filemode='w'
                            )

        self.titles = []
        self.executor = ThreadPoolExecutor(max_workers=24)

        if test_mode:
            self.parse_month(
                'https://en.wikipedia.org/wiki/Category:March_1992_events')
            return

        ts = datetime.now()
        self.executor = ThreadPoolExecutor(max_workers=32)
        urls = [template_url.format(month, year)
                for year in range(start_year, end_year + 1)
                for month in calendar.month_name[1:]]
        for url in urls:
            self.executor.submit(self.parse_month, url)
        self.executor.shutdown(wait=True)
        with open('wiki_events.json', 'w', encoding='utf8') as outfile:
            json.dump(self.titles, outfile)
            delta = datetime.now() - ts
            logging.critical('dumped %s, took %s (%d titles)', outfile.name, str(delta), len(self.titles))
        logging.critical('finished.')

    def parse_month(self, url):
        response = requests.get(url, headers=self.HEADERS)
        if response.status_code != 200:
            logging.error('Got status code %i from %s', response.status_code, url)
            return
        c = response.content
        soup = BeautifulSoup(c, 'lxml')
        for a in soup.select('div.mw-category a'):
            title = ftfy.fix_text(a.get('title'))
            self.titles.append(title)


if __name__ == '__main__':
    WikiEventsCrawler()
