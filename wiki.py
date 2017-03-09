import bz2
import collections
import os
import re
from lxml import etree
import numpy as np
from attrdict import AttrDict
import tensorflow as tf
import lazy_property

TOKEN_REGEX = re.compile(r'[A-Za-z]+|[!?.:,()]')

class Wikipedia:

    def __init__(self, url, cache_dir, vocabulary_size=10000):
        self._cache_dir = os.path.expanduser(cache_dir)
        self._pages_path = os.path.join(self._cache_dir, 'pages.bz2')
        self._vocabulary_path = os.path.join(self._cache_dir, 'vocabulary.bz2')
        if not os.path.isfile(self._pages_path):
            print('Read pages')
            self._read_pages(url)
        if not os.path.isfile(self._vocabulary_path):
            print('Build vocabulary')
            self._build_vocabulary(vocabulary_size)
        with bz2.open(self._vocabulary_path, 'rt') as vocabulary:
            print('Read vocabulary')
            self._vocabulary = [x.strip() for x in vocabulary]
        self._indices = {x: i for i, x in enumerate(self._vocabulary)}

    def __iter__(self):
        with bz2.open(self._pages_path, 'rt') as pages:
            for page in pages:
                words = page.strip().split()
                words = [self.encode(x) for x in words]
                yield words

    @property
    def vocabulary_size(self):
        return len(self._vocabulary)

    def encode(self, word):
        """Get the vocabulary index of a string word."""
        return self._indices.get(word, 0)

    def decode(self, index):
        """Get back the string word from a vocabulary index."""
        return self._vocabulary[index]

    def _read_pages(self, url):
        """
        Extract plain words from a Wikipedia dump and store them to the pages
        file. Each page will be a line with words separated by spaces.
        """
        wikipedia_path = download(url, self._cache_dir)
        with bz2.open(wikipedia_path) as wikipedia, \
                bz2.open(self._pages_path, 'wt') as pages:
            for _, element in etree.iterparse(wikipedia, tag='{*}page'):
                if element.find('./{*}redirect') is not None:
                    continue
                page = element.findtext('./{*}revision/{*}text')
                words = self._tokenize(page)
                pages.write(' '.join(words) + '\n')
                element.clear()

    def _build_vocabulary(self, vocabulary_size):
        """
        Count words in the pages file and write a list of the most frequent
        words to the vocabulary file.
        """
        counter = collections.Counter()
        with bz2.open(self._pages_path, 'rt') as pages:
            for page in pages:
                words = page.strip().split()
                counter.update(words)
        common = ['<unk>'] + counter.most_common(vocabulary_size - 1)
        common = [x[0] for x in common]
        with bz2.open(self._vocabulary_path, 'wt') as vocabulary:
            for word in common:
                vocabulary.write(word + '\n')

    @classmethod
    def _tokenize(cls, page):
        words = cls.TOKEN_REGEX.findall(page)
        words = [x.lower() for x in words]
        return words

def skipgrams(pages, max_context):
    """Form training pairs according to the skip-gram model."""
    for words in pages:
        for index, current in enumerate(words):
            context = random.randint(1, max_context)
            for target in words[max(0, index - context): index]:
                yield current, target
            for target in words[index + 1: index + context + 1]:
                yield current, target


def batched(iterator, batch_size):
    """Group a numerical stream into batches and yield them as Numpy arrays."""
    while True:
        data = np.zeros(batch_size)
        target = np.zeros(batch_size)
        for index in range(batch_size):
            data[index], target[index] = next(iterator)
        yield data, target



class EmbeddingModel:

    def __init__(self, data, target, params):
        self.data = data
        self.target = target
        self.params = params
        self.embeddings
        self.cost
        self.optimize

    @lazy_property.LazyProperty
    def embeddings(self):
        initial = tf.random_uniform(
            [self.params.vocabulary_size, self.params.embedding_size],
            -1.0, 1.0)
        return tf.Variable(initial)

    @lazy_property.LazyProperty
    def optimize(self):
        optimizer = tf.train.MomentumOptimizer(
            self.params.learning_rate, self.params.momentum)
        return optimizer.minimize(self.cost)

    @lazy_property.LazyProperty
    def cost(self):
        embedded = tf.nn.embedding_lookup(self.embeddings, self.data)
        weight = tf.Variable(tf.truncated_normal(
            [self.params.vocabulary_size, self.params.embedding_size],
            stddev=1.0 / self.params.embedding_size ** 0.5))
        bias = tf.Variable(tf.zeros([self.params.vocabulary_size]))
        target = tf.expand_dims(self.target, 1)
        return tf.reduce_mean(tf.nn.nce_loss(
            weight, bias, embedded, target,
            self.params.contrastive_examples,
            self.params.vocabulary_size))

params = AttrDict(
    vocabulary_size=10000,
    max_context=10,
    embedding_size=200,
    contrastive_examples=100,
    learning_rate=0.5,
    momentum=0.5,
    batch_size=1000,
)

data = tf.placeholder(tf.int32, [None])
target = tf.placeholder(tf.int32, [None])
model = EmbeddingModel(data, target, params)

corpus = Wikipedia(
    'https://dumps.wikimedia.org/enwiki/20160501/' \
    'enwiki-20160501-pages-meta-current1.xml-p000000010p000030303.bz2',
    'tmp/wikipedia',
    params.vocabulary_size)
examples = skipgrams(corpus, params.max_context)
batches = batched(examples, params.batch_size)

sess = tf.Session()
sess.run(tf.initialize_all_variables())
average = collections.deque(maxlen=100)
for index, batch in enumerate(batches):
    feed_dict = {data: batch[0], target: batch[1]}
    cost, _ = sess.run([model.cost, model.optimize], feed_dict)
    average.append(cost)
    print('{}: {:5.1f}'.format(index + 1, sum(average) / len(average)))

embeddings = sess.run(model.embeddings)
np.save('tmp/wikipedia/embeddings.npy', embeddings)