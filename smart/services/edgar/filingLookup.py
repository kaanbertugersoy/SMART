import requests
from bs4 import BeautifulSoup


BASE_URL = "https://www.sec.gov/cgi-bin/browse-edgar"
QUERY_LIMIT = 999


def search_filing(cik, type='10-Q'):
    params, headers = build_req_params(cik, type, start_date='2001-01-01')

    response = requests.get(BASE_URL, headers=headers, params=params)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, 'xml')
    entries = soup.find_all('entry')
    archive = f"https://www.sec.gov/Archives/edgar/data/{cik}/accession_number/file.htm"

    filings = []
    for entry in entries:
        title = entry.find('title')
        updated = entry.find('updated')
        accession_number = entry.find('accession-number')
        category = entry.find('category')
        link = entry.find('link', {'rel': 'alternate'})

        if title and updated and accession_number and category and link:
            filing = {
                'accession_number': accession_number.text,
                'filing_date': updated.text,
                'form_type': category['term'],
                'link': link['href']
            }
            filings.append(filing)
        else:
            print("Missing expected fields in entry, skipping...")

    return filings


def build_req_params(cik, type, start_date):
    params = {
        'action': 'getcompany',
        'CIK': cik,
        'type': type,
        'datea': start_date,
        'dateb': '',
        'owner': 'exclude',
        'count': QUERY_LIMIT,
        'output': 'atom'
    }

    headers = {
        'User-Agent': 'Kaan Ersoy (kaanersoy@windowslive.com)',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8'
    }

    return params, headers
