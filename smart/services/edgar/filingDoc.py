import sec_parser as sp

# Initialize the parser
url = "https://www.sec.gov/Archives/edgar/data/320193/000032019324000069/aapl-20240330.htm"
elements: list = sp.Edgar10QParser().parse(url)

print(sp.render(elements))
# Specify the URL of the SEC filing

# Parse the SEC filing and extract financial data
financial_data = parser.parse(url)

# Display the extracted financial data
for table_name, df in financial_data.items():
    print(f"Table: {table_name}")
    print(df)  # This will print each DataFrame extracted

# Example: Access a specific table, like the Balance Sheet
balance_sheet = financial_data.get('Balance Sheet', None)
if balance_sheet is not None:
    print("Balance Sheet:")
    print(balance_sheet)
