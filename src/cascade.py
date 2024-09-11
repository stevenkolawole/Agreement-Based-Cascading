import time
import re
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

from together import Together
from openai import OpenAI

from rich import print as rprint


class API:
    required_attributes = ["API_KEY"]

    def __init__(self):
        client = self.Provider(api_key=self.API_KEY)



class TogetherAPI(API):
    API_KEY = os.f