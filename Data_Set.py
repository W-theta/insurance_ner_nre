from Entity import Document
from Prepare_sents import SentenceExtractor
from collections import Counter


class DataSet:
    def __init__(self, file_names: str, root_path: str, word2index=None, vocab_size=2000):
        self.file_names = file_names
        self.vocab_size = vocab_size

        docs = []
        for f in file_names:
            d = Document(f, root_path)
            docs.append(d)

        self._docs = docs
        self._word2index = word2index

    @property
    def word2idx(self):
        if self._word2index is None:
            return self.__build_vocab_dic()
        else:
            return self._word2index

    @property
    def docs(self):
        return self._docs

    def __build_vocab_dic(self):
        vocab_dic = {}
        vocab_size = self.vocab_size
        counter = Counter()
        for doc in self._docs:
            for single_c in doc.text:
                counter[single_c] += 1
        vocab_dic['_padding'] = 0
        vocab_dic['_unk'] = 1
        if vocab_size > 0:
            if (vocab_size - 2) <= len(counter):
                most_common_count = vocab_size - 2
            else:
                most_common_count = len(counter)
        else:
            most_common_count = len(counter)
        for idx, (char, _) in enumerate(counter.most_common(most_common_count)):
            vocab_dic[char] = idx + 2
        return vocab_dic

    def __vectorize(self):
        new_docs = []
        for doc in self._docs:
            doc2idx = []
            for c in doc.text:
                if c in self.word2idx.keys():
                    doc2idx.append(self.word2idx[c])
                else:
                    doc2idx.append(self.word2idx["_unk"])
            doc.text = doc2idx
            new_docs.append(doc)
        return new_docs

    def get_sents_set(self, window_size, padding_size):
        docs = self.__vectorize()
        sents = []
        for doc in docs:
            sent_extractor = SentenceExtractor(window_size, padding_size, doc)
            sentences = sent_extractor.extract()
            sents.append(sentences)
        return sents
