import csv
import re
from typing import Dict, Tuple, Any

import tqdm
import numpy as np
import pandas as pd
import nltk as nl


# constants needed for normalization
non_alphas = re.compile(u'[^A-Za-z<>]+')
cont_patterns = [
    ('(W|w)on\'t', 'will not'),
    ('(C|c)an\'t', 'can not'),
    ('(I|i)\'m', 'i am'),
    ('(A|a)in\'t', 'is not'),
    ('(\w+)\'ll', '\g<1> will'),
    ('(\w+)n\'t', '\g<1> not'),
    ('(\w+)\'ve', '\g<1> have'),
    ('(\w+)\'s', '\g<1> is'),
    ('(\w+)\'re', '\g<1> are'),
    ('(\w+)\'d', '\g<1> would'),
]
patterns = [(re.compile(regex), repl) for (regex, repl) in cont_patterns]


class PreProcessor(object):

    def __init__(self):
        self.stemmer = nl.PorterStemmer()

    def load_embedding(self, path_to_embedding: str) -> Tuple[Any, Any]:

        word2vec, word2id = {}, {}
        count = 0

        with open(path_to_embedding, "rb") as file:
            for line in file:
                l = line.decode().split()
                word = l[0]
                word2vec[word] = np.array(l[1:]).astype(np.float)
                word2id[word] = count
                count += 1

        return word2vec, word2id

    def load_and_prep_datasets(self, trainset: str, testset: str, n_train: int, n_val: int):
        """
        Takes the train and test set as CSV along with the number of train and value samples.
        :param trainset: str pointing to CSV file location.
        :param testset: str pointing to CSV file location.
        :param n_train: int
        :param n_val: int
        :return: Pandas(train), values, Pandas(test)
        """

        df_train = pd.read_csv(filepath_or_buffer=trainset, encoding="latin-1", header=0,
                               names=["polarity", "id", "date", "query", "user", "tweet"])
        df_test = pd.read_csv(filepath_or_buffer=testset, encoding="latin-1", header=0,
                              names=["polarity", "id", "date", "query", "user", "tweet"])

        # Exclude neutral tweets (0 = negative, 2 = neutral, 4 = positive)
        df_train = df_train[df_train.polarity != 2]
        df_train.polarity = df_train.polarity // 4

        # Keep polarity and tweet only
        df_train = df_train[["polarity", "tweet"]]
        df_train = df_train.sample(frac=1).reset_index(drop=True)
        val = df_train.iloc[:n_val]
        df_train = df_train.iloc[n_val : n_val + n_train]

        # Prepare the test set
        df_test = df_test[df_test.polarity != 2]
        df_test.polarity = df_test.polarity // 4
        df_test = df_test[["polarity", "tweet"]]

        return df_train, val, df_test

    def prep_hashtags(self, x: str):
        """
        Normalizes hashtags and marks them.
        :param x: str
        :return: marked hashtag str.
        """
        s = x.group()
        if s.upper() == s:
            return "<hashtag>" + s.lower() + "<allcaps>"
        else:
            return "<hashtag>" + " ".join(re.findall("[A-Z]*[^A-Z]*", s)).lower()

    def to_lower(self, x: str):
        return x.group().lower() + "<allcaps>"

    def prep_tweet_text(self, x: str):
        """
        Following the GloVe preprocessing
        :param x: str
        :return: GloVe processed str
        """
        # for tagging urls
        text = re.sub('(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/|www\.){1}[A-Za-z0-9.\/\\]+[]*', ' <url> ',x)
        # for tagging users
        text = re.sub("\[\[User(.*)\|", ' <user> ', text)
        text = re.sub('@[^\s]+', ' <user> ', text)
        # for tagging numbers
        text = re.sub("[-+]?[.\d]*[\d]+[:,.\d]*", " <number> ", text)
        # for tagging emojis
        eyes = "[8:=;]"
        nose = "['`\-]?"
        text = re.sub("<3", ' <heart> ', text)
        text = re.sub(eyes + nose + "[Dd)]", ' <smile> ', text)
        text = re.sub("[(d]" + nose + eyes, ' <smile> ', text)
        text = re.sub(eyes + nose + "p", ' <lolface> ', text)
        text = re.sub(eyes + nose + "\(", ' <sadface> ', text)
        text = re.sub("\)" + nose + eyes, ' <sadface> ', text)
        text = re.sub(eyes + nose + "[/|l*]", ' <neutralface> ', text)
        # split / from words
        text = re.sub("/", " / ", text)
        # remove punctuation
        text = re.sub('[.?!:;,()*]+', ' ', text)
        # tag and process hashtags
        text = re.sub(r'#([^\s]+)', self.prep_hashtags, text)
        # for tagging allcaps words
        text = re.sub("([^a-z0-9()<>' `\-]){2,}", self.to_lower, text)
        # find elongations in words ('hellooooo' -> 'hello <elong>')
        pattern = re.compile(r"(.)\1{2,}")
        text = pattern.sub(r"\1" + " <elong> ", text)
        return text

    def normalize_tweet(self, x: str):
        clean = x.lower()
        clean = clean.replace('\n', ' ')
        clean = clean.replace('\t', ' ')
        clean = clean.replace('\b', ' ')
        clean = clean.replace('\r', ' ')
        for (pattern, repl) in patterns:
            clean = re.sub(pattern, repl, clean)
        return u' '.join([y for y in non_alphas.sub(' ', clean).strip().split(' ')])

    def generate_vocab(self, df: pd.DataFrame):
        """
        Generates a set of unique words found in tweets.
        :param df: Ingests a pandas dataframe of shape (polarity, tweet)
        :return: Set (vocabulary)
        """
        vocab = set()
        tweets = df.tweet.values

        for tweet in tqdm.tqdm(tweets):
            words = self.normalize_tweet(self.prep_tweet_text(tweet)).split(" ")

            for word in words:
                vocab.add(self.stemmer.stem(word))

        return vocab

    def to_vector(self, tweet, w2v):
        """
        Translate a preprocessed tweet to an embedding vector.
        :param tweet: pre-processed tweet
        :param w2v: translator from word to vector
        :return: vector of len 200
        """
        return np.mean([w2v.get(self.stemmer.stem(bit), np.zeros(shape=(200,))) for bit in tweet.split(" ")], 0)

    def vectorize_data(self, df: pd.DataFrame, w2v):

        print("Keys:", df.keys())

        X = np.stack(
            df.tweet.apply(self.prep_tweet_text).apply(self.normalize_tweet).apply(lambda x: self.to_vector(x, w2v))
        )
        Y = df.polarity.values.reshape(-1, 1)

        return X, Y

    def embed_vocab(self, voc, w2v: Dict) -> Tuple[Any, Any]:

        keys = w2v.keys() & voc
        restricted_w2id = dict(zip(keys, range(len(keys))))
        id2restricted_w = {v: k for k, v in restricted_w2id.items()}

        # Tokenize vocabulary
        mx = np.array([w2v[id2restricted_w[idx]]] for idx in range(len(id2restricted_w)))
        mx = np.vstack((mx, mx.mean(0)))
        mx = np.vstack((mx, np.zeros((1, mx.shape[1]))))

        return restricted_w2id, mx

    def tokenize_tweet(self, x, w2id, pad_len):
        bits = x.split(" ")
        return np.array([w2id.get(self.stemmer.stem(t), len(w2id)) for t in bits[:min(pad_len, len(bits))]] +
                        max(pad_len - len(bits), 0) * [len(w2id) + 1])

    def tokenize_data(self, df, w2id, pad_len=40):
        X = np.stack(
            df.tweet.apply(self.prep_tweet_text).apply(self.normalize_tweet).apply(lambda x: self.tokenize_tweet(x, w2id, pad_len))
        )
        Y = df.polarity.values.reshape(-1, 1)

        return X, Y