from camel_tools.utils.charsets import AR_CHARSET, UNICODE_PUNCT_CHARSET
from nltk.lm.vocabulary import Vocabulary
import string

pad_token = "<PAD>"
bos_token = "<BOS>"
eos_token = "<EOS>"

CHARSET = (
    sorted(AR_CHARSET)
    + sorted(string.printable)
    + sorted(UNICODE_PUNCT_CHARSET)
    + [pad_token, bos_token, eos_token]
)

ARABIC_VOCABULARY = Vocabulary(CHARSET)
