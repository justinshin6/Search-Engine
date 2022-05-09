import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


class TextProcessor:
    ''' Class for stemming and tokenizing words and checking for stop words '''

    def __init__(self):
        ''' Constructor for the TextProcessor class '''

        self.STOP_WORDS = set(stopwords.words("english"))
        self.stemmer = PorterStemmer()
        self.n_regex = '''\[\[[^\[]+?\]\]|[a-zA-Z0-9]+'[a-zA-Z0-9]+|[a-zA-Z0-9]+'''


    def stem_word(self, word: str) -> str:
        ''' 
        Produces the stem for a given word

        Parameters:
        word (str) -- word to stem

        Returns:
        (str) -- the stemmed word
        '''

        return self.stemmer.stem(word)


    def tokenize(self, text: str) -> "list[str]":
        '''
        Tokenizes a string of text

        Parameters:
        text (str) -- text to tokenize

        Returns:
        (str) -- tokenized text
        '''

        return re.findall(self.n_regex, text.lower())


    def is_link(self, token: str) -> bool:
        '''
        Determines whether a given token is a link

        Parameters:
        token (str) -- token to check

        Returns:
        (bool) -- true if given token is a link, false otherwise
        '''

        return token[0:2] == "[[" and token[-2:] == "]]"

    
    def is_stop_word(self, token: str) -> bool:
        '''
        Determines whether a given token is a stop word

        Parameters:
        token (str) -- token to check

        Returns:
        (bool) -- true if the given token is a stop word, false otherwise
        '''

        return token in self.STOP_WORDS