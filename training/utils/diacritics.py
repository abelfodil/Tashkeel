import re
import unicodedata

# Source: https://github.com/linuxscout/pyarabic/blob/master/pyarabic/araby.py#L290
DIACRITICS = [chr(x) for x in range(0x0600, 0x06ff) if unicodedata.category(chr(x)) == "Mn"]
DIACRITICS_PATTERN = re.compile(f"[{u''.join(DIACRITICS)}]")

def strip_diacritics(text):
    return re.sub(DIACRITICS_PATTERN, '', text)
