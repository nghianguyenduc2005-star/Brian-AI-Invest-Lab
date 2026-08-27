from ai.client import ask
from ai.prompts import client_message

def make_client_message(action, symbol, price, extra=""):
    return ask(client_message(action, symbol, price, extra))
