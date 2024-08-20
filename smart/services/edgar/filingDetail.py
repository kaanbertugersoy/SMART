from bs4 import BeautifulSoup
import requests


def find_document_link(filing_link):
    headers = build_req_params()

    response = requests.get(filing_link, headers=headers)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, 'html.parser')

    table = soup.find('table', {'summary': 'Document Format Files'})

    document_links = []
    if table:
        rows = table.find_all('tr')
        for row in rows[1:]:
            cols = row.find_all('td')
            if len(cols) >= 3:
                description = cols[1].text.strip()
                link_tag = cols[2].find('a')
                if link_tag:
                    document_link = 'https://www.sec.gov' + link_tag['href']
                    if document_link.startswith('https://www.sec.gov/ix?doc='):
                        document_link = document_link.replace('/ix?doc=/', '/')
                    document_links.append((description, document_link))

    return document_links


def build_req_params():
    headers = {
        'User-Agent': 'Kaan Ersoy (kaanersoy@windowslive.com)',
    }

    return headers
