from camel_tools.utils.charsets import (
    AR_CHARSET,
    UNICODE_LETTER_MARK_NUMBER_CHARSET,
    UNICODE_PUNCT_SYMBOL_CHARSET,
)
from nltk.lm.vocabulary import Vocabulary

unk_token = "<unk>"
pad_token = "<pad>"
bos_token = "<bos>"
eos_token = "<eos>"

CHARSET = (
    [unk_token, pad_token, bos_token, eos_token]
    + sorted(AR_CHARSET)
    + sorted(UNICODE_LETTER_MARK_NUMBER_CHARSET)
    + sorted(UNICODE_PUNCT_SYMBOL_CHARSET)
)

ARABIC_VOCABULARY = Vocabulary(CHARSET)
