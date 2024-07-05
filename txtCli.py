from twilio.rest import Client

# Test code for message functionality (UNNECESSARY)

account_sid = 'AC380de7d47e0b4b7d0f774ab1d5ea5ccb'
auth_token = '019af30d1a265b9324412e721685209f'
client = Client(account_sid, auth_token)


def send_sms_report():
    message = client.messages.create(
        from_='+15018177916',
        body=(
            "SMART Execution Report \n\n"
            "| Stock |  Action | Price  |\n"
            "|-------|---------|--------|\n"
            "| AAPL  |   Buy   | $150.00|\n"
            "| MSFT  |   Sell  | $280.00|\n"
            "| GOOGL |   Buy   | $2500.00|\n"
            "| AMZN  |   Sell  | $3500.00|"
        ),
        to='+905305701693'
    )

    print(message.sid)
