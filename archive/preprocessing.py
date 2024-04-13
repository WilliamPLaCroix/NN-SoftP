import re

import pandas as pd


def filter_unicodes(text: str):
    if pd.notna(text):
        text = re.sub("\n", " ", text)
        regex_urls = r"https?\S+|\S*\.com\S*"
        text = re.sub(regex_urls, "", text)
        if text.strip():
            regex_han_ascii = re.compile(r"[^\u4E00-\u9FFF\u3000-\u303F\u0000-\u007F\u2000-\u206F\uFF00-\uFF65]")
            text = regex_han_ascii.sub("", text)
            if not text.strip():
                text = pd.NA
        else:
            text = pd.NA
    return text
    