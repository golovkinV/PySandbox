from bs4 import BeautifulSoup
from decimal import Decimal

def calculate_amount(currency, amount):
    params = get_currency_params(currency)
    nominal, value = params[0], params[1]
    result = Decimal(amount) / value * nominal
    return quantize_result(result)


def calculate_currency(amount, currency_from, currency_to):
    params = get_currency_params(currency_to)
    nominal, value = params[0], params[1]
    result = calculate_amount(currency_from, Decimal(amount)) / value * nominal
    return quantize_result(result)


def get_currency_params(currency):
    return int(currency.find('nominal').text), Decimal(currency.find('value').text.replace(',', '.'))


def quantize_result(result):
    return Decimal(result).quantize(Decimal("1.0000"))


def convert(amount, cur_from, cur_to, date, requests):
    response = requests.get(url=f"https://www.cbr.ru/scripts/XML_daily.asp?date_req={date}")
    if cur_from == "RUR" and cur_to == "RUR":
        return Decimal(amount).quantize(Decimal("1.0000"))
    elif cur_from == "RUR" or cur_to == "RUR":
        cur = cur_to if cur_from == "RUR" else cur_from
        bs = BeautifulSoup(markup=response.text, features="lxml")
        currency = bs.find(string=cur).parent.parent
        return calculate_amount(currency, amount)
    else:
        bs = BeautifulSoup(markup=response.text, features='lxml')
        currency_from = bs.find(string=cur_from).parent.parent
        currency_to = bs.find(string=cur_to).parent.parent
        return calculate_currency(amount, currency_from, currency_to)
