from Utils import scan_files, load_file

import numpy as np
import gensim
from gensim.models.word2vec import LineSentence
from gensim.models import Word2Vec
import multiprocessing

from keras.optimizers import Adam
from keras import Input
from keras.layers import Embedding, GRU, LSTM, Bidirectional,Dropout
from keras_contrib.layers import CRF
from keras.models import Model


# 字-Vec的训练器
class Char2VecTrainer:
    def __init__(self, root: str, w2v_file_path: str):
        self.root = root
        self._w2v_file_path = w2v_file_path

    def prepare_data(self):
        with open(self._w2v_file_path, 'w', encoding='utf-8') as f:
            file_names = scan_files(self.root)
            for name in file_names:
                data = load_file(self.root, name, "txt")
                data = " ".join(data)
                f.write(data)
                f.write("\n")
        return

    def train(self, output: str, emb_size=128):
        model = Word2Vec(LineSentence(self._w2v_file_path), size=emb_size, window=5, sg=0, hs=0, negative=3,
                         workers=multiprocessing.cpu_count())
        model.save(output)
        return

    def load(self):
        w2v_model = Word2Vec.load(self._w2v_file_path)
        print('w2v的模型维度是：{}'.format(w2v_model.wv.vector_size))
        print('w2v的模型的词表总长是：{}'.format(len(w2v_model.wv.vocab)))
        return w2v_model


# NER的BaseLine训练器
class BiLstmCrfTrainer:
    def __init__(self, category_count, seq_len, vocab_size, emb_matrix=None, lstm_units=256, optimizer=Adam()):
        self.category_count = category_count
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.emb_matrix = emb_matrix
        self.lstm_units = lstm_units
        self.optimizer = optimizer

    def build(self):
        if self.emb_matrix is not None:
            embedding = Embedding(input_dim=self.vocab_size,
                                  output_dim=self.emb_matrix.shape[1],
                                  weights=[self.emb_matrix],
                                  trainable=False)
        else:
            embedding = Embedding(input_dim=self.vocab_size, output_dim=128, trainable=True)

        model_input = Input(shape=(self.seq_len,), dtype="int32")
        embedding = embedding(model_input)
        dropout = Dropout(0.5)(embedding)
        lstm = LSTM(self.lstm_units, return_sequences=True)
        bi_lstm = Bidirectional(lstm)(dropout)
        crf = CRF(self.category_count, sparse_target=True)
        output = crf(bi_lstm)

        model = Model(model_input, output)
        model.summary()
        model.compile(optimizer=self.optimizer,
                      loss=crf.loss_function,
                      metrics=[crf.accuracy])
        return model

        # NER的Attention层替换训练器


class BiLstmLanTrainer:
    pass


# NER的原型网络小样本训练器
class ProtoNetTrainer:
    pass


# 实体关系抽取训练器
class RelationExtractorTrainer:
    pass


# 实体+关系的联合抽取训练器
class JointExtractorTrainer:
    pass
