from nltk.corpus import stopwords       # lista stopwords
import unicodedata      # Remocao de acentos
import string
from nltk.stem.porter import *  # porterStemmer

def ToLowerCase(text):
    return text.lower()

def KeepOnlySome(text, some=[], sep=' '):
    tokens = text.split(sep)
    for i in range(len(tokens)):
        tokens[i] = ''.join( ( c for c in list(tokens[i]) if c in some) )
    
    return sep.join(tokens)

def RemoveStopWords(text, stopwordsList=[], sep=' '):
    tokens = text.split(sep)
    return [tok for tok in tokens if tok not in stopwordsList]

def StripAccents(text, IsOn=True):
    return (unicodedata.normalize('NFKD', text)
           .encode('ISO-8859-1', 'ignore')
           .decode('ISO-8859-1')
    ) if IsOn else text

def RemoveIfNotInList(text, toRemove=[]):
    return RemoveStopWords(text, stopwordsList=toRemove)

def RemovePunctuation(text, toRemove = string.punctuation):
    return RemoveIfNotInList(text, toRemove = toRemove)

def RemoveAlphaNum(text, toRemove = string.digits):
    return RemoveIfNotInList(text, toRemove = toRemove )

def RemoveAllButAscii(text):
    return KeepOnlySome(text, some=string.ascii_letters)
def RemoveAllButAsciiLower(text):
    return KeepOnlySome(text, some=string.ascii_lowercase)
_stm = PorterStemmer()
def StemmizePorter(text):
    return ' '.join(_stm.stem(c) for c in text.split())

# Passos para limpeza de texto:
# 1 - Deixar tudo minusculo
# 2 - remover acentos
# 3 - remover stop words
# 4 - remover pontuação
# 0+ : Especifico.... steeming, lemmatization, ...
engStopWords = stopwords.words("english")

def pre_process_lower(text):
    text = ToLowerCase(text);    print("==lower==> ", text)
    text = StripAccents(text);    print("==no_accents==> ", text)
    text = RemoveAllButAsciiLower(text);    print("==only_ascii==> ", text)
    text = ' '.join( RemoveStopWords( text, engStopWords ))
    return text

def main():
    textRaw = "Some people arén'T the best ones to ask for an advice"
    text = pre_process_lower(textRaw)
    print(text)
if __name__ == '__main__':
    main()