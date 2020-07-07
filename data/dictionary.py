from collections import Counter


class Dictionary:
    """ A class for storing indices of words
    and languages."""

    def __init__(self):
        self.tkn2idx = {}
        self.idx2tkn = []
        self.counter = Counter()
        self.total = 0
        self.lang2idx = {}
        self.idx2lang = []
        self.counter_lang = Counter()
        self.total_lang = 0

    def add_tkn(self, tkn: str) -> int:
        """
        Add token to index if it does not already exist.

        Parameters
        ----------
        tkn : str
            A token.
        Returns
        -------
        int :
            The integral index of the token.
        """
        if tkn not in self.tkn2idx:
            self.idx2tkn.append(tkn)
            self.tkn2idx[tkn] = len(self.idx2tkn) - 1

        token_id = self.tkn2idx[tkn]
        self.counter[token_id] += 1
        self.total += 1

        return self.tkn2idx[tkn]

    def add_lang(self, lang: str) -> int:
        """
        Add a language to index if it does not already exist.
        Parameters
        ----------
        lang : str
            The language.
        Returns
        -------
        int :
            The integral index of the language
        """
        if lang not in self.lang2idx:
            self.idx2lang.append(lang)
            self.lang2idx[lang] = len(self.idx2lang) - 1
            self.total_lang += 1

        lang_id = self.lang2idx[lang]
        self.counter_lang[lang_id] += 1

        return self.lang2idx[lang]

    def __len__(self):
        return len(self.idx2tkn)
