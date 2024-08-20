import requests
from bs4 import BeautifulSoup
import pandas as pd
import lxml.etree as ET
import time

from tqdm import tqdm

from smart.services.edgar.filingLookup import search_filing
from smart.services.edgar.filingDetail import find_document_link
# from smart.services.edgar.filingDoc import extract_filing_data


cik = '0000320193'  # CIK for Apple Inc.


def edgarExtractionTest():
    filings = search_filing(cik, type='10-Q')

    filing_links = [filing["link"] for filing in filings]

    print(filing_links[0])
    document_link = find_document_link(filing_links[0])
    print(document_link)
    # main_document_links = []
    # for link in tqdm(filing_links, desc='Finding 10-Q document links', unit='link'):
    #     document_links = find_document_link(link)
    #     for description, doc_link in document_links:
    #         if '10-Q' in description:
    #             main_document_links.append(doc_link)
    #             break

    # print(main_document_links[0])

    # df = extract_filing_data(main_document_links[0])
    # print(df)

# # Example usage
# if __name__ == "__main__":
#     cik = '0000320193'  # CIK for Apple Inc.
#     try:
#         filings = get_10q_filings(cik)
#         # Fetch details for the first filing
#         if filings:
#             first_filing = filings[0]
#             document_links = get_document_links(first_filing['link'])

#             # Find the main 10-Q document link (usually the first one or containing "10-Q" in description)
#             main_document_link = None
#             for description, doc_link in document_links:
#                 if '10-Q' in description:
#                     main_document_link = doc_link
#                     break

#             if main_document_link:
#                 details_df = extract_10q_details_xbrl(main_document_link)
#                 for idx, df in enumerate(details_df):
#                     print(f"Table {idx}:\n{df.head()}\n")
#                 balance_sheet_df = details_df[1]
#                 print("Balance Sheet:\n", balance_sheet_df)
#             else:
#                 print("Main 10-Q document link not found.")
#     except requests.exceptions.HTTPError as e:
#         print(f"HTTP error occurred: {e}")
#     except Exception as e:
#         print(f"An error occurred: {e}")
#     finally:
#         time.sleep(1)  # Sleep for a second to comply with SEC's rate limiting
