from common.Utils import scan_files, load_file

from gensim.models.word2vec import LineSentence
from gensim.models import Word2Vec
import multiprocessing

from keras.optimizers import Adam
from keras import Input
from keras.layers import Embedding, LSTM, Bidirectional, Dropout, Layer, Concatenate
from keras_contrib.layers import CRF
from keras.models import Model

from keras import backend as K
from keras import initializers, regularizers, constraints

import kashgari

kashgari.config.use_cudnn_cell = True
from kashgari.embeddings import BERTEmbedding
from kashgari.tasks.labeling import BiLSTM_CRF_Model


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


class NNTrainer:
    def __init__(self, category_count, seq_len, vocab_size, lstm_units, emb_matrix=None, optimizer=Adam()):
        self.category_count = category_count
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.emb_matrix = emb_matrix
        self.lstm_units = lstm_units
        self.optimizer = optimizer

    def build(self):
        pass


# NER的BaseLine训练器
class BiLstmCrfTrainer(NNTrainer):

    def __init__(self, category_count, seq_len, vocab_size, lstm_units, emb_matrix=None, optimizer=Adam()):
        super().__init__(category_count, seq_len, vocab_size, lstm_units, emb_matrix, optimizer)

    def build(self):
        if self.emb_matrix is not None:
            embedding = Embedding(input_dim=self.vocab_size,
                                  output_dim=self.emb_matrix.shape[1],
                                  weights=[self.emb_matrix],
                                  trainable=False)
        else:
            embedding = Embedding(input_dim=self.vocab_size, output_dim=128, trainable=True
                                  )  # mask_zero=True 这里给embedding的zero做mask

        model_input = Input(shape=(self.seq_len,), dtype="int32")
        embedding = embedding(model_input)
        dropout = Dropout(0.5)(embedding)
        lstm = LSTM(self.lstm_units, return_sequences=True)
        bi_lstm = Bidirectional(lstm)(dropout)
        bi_lstm = Dropout(0.5)(bi_lstm)
        crf = CRF(self.category_count, sparse_target=True)
        output = crf(bi_lstm)

        model = Model(model_input, output)
        model.summary()
        model.compile(optimizer=self.optimizer,
                      loss=crf.loss_function,
                      metrics=[crf.accuracy])
        return model

        # NER的Attention层替换训练器


class BiLstm_Lan_Trainer(NNTrainer):
    def __init__(self, category_count, seq_len, vocab_size, lstm_units, emb_matrix=None, optimizer=Adam()):
        super().__init__(category_count, seq_len, vocab_size, lstm_units, emb_matrix, optimizer)

    def build(self):
        if self.emb_matrix is not None:
            embedding = Embedding(input_dim=self.vocab_size,
                                  output_dim=self.emb_matrix.shape[1],
                                  weights=[self.emb_matrix],
                                  trainable=False)
        else:
            embedding = Embedding(input_dim=self.vocab_size, output_dim=128, trainable=True
                                  )  # mask_zero=True 这里给embedding的zero做mask

        model_input = Input(shape=(self.seq_len,), dtype="int32")
        embedding = embedding(model_input)
        x = Dropout(0.5)(embedding)

        for idx, param in enumerate(self.lstm_units):
            lstm = LSTM(param, return_sequences=True)
            if idx == len(self.lstm_units) - 1:
                # 表示是最后一层：
                x = Bidirectional(lstm)(x)
                x = Attention(self.category_count, is_last_layer=True)(x)
            else:
                x_1 = Bidirectional(lstm)(x)
                x_2 = Attention(self.category_count, is_last_layer=False)(x_1)
                x = Concatenate()([x_1, x_2])

        model = Model(model_input, x)

        model.summary()
        model.compile(optimizer=self.optimizer,
                      loss="sparse_categorical_crossentropy",
                      metrics=["acc"])
        return model


class Attention(Layer):
    """
    Describe:
        Input : (samples,steps,fetures)
        Output:(samples,steps,fetures)
    """

    def __init__(self, categeory_count, is_last_layer=False, W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None, **kwargs):
        self.categeroy_count = categeory_count
        self.is_last_layer = is_last_layer
        self.W_regularizer = regularizers.get(W_regularizer)
        self.W_constraint = constraints.get(W_constraint)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.b_constraint = regularizers.get(b_constraint)

        self.init = initializers.get("glorot_uniform")

        self.steps_dim = 0
        self.features_dim = 0

        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(
            input_shape) == 3, "Att's input shape is {}.You'd turn shape into (samples,steps,features)".format(
            input_shape)

        self.steps_dim = input_shape[1]
        self.features_dim = input_shape[2]

        self.WQ = self.add_weight(name="WQ", shape=(self.features_dim, self.features_dim),
                                  initializer=self.init, trainable=True)
        self.label_emb = self.add_weight(name="Label", shape=(self.categeroy_count, self.features_dim),
                                         initializer=self.init, trainable=True)
        self.WK = self.add_weight(name="WK", shape=(self.features_dim, self.features_dim),
                                  initializer=self.init, trainable=True)
        if self.is_last_layer is False:
            self.WV = self.add_weight(name="WV", shape=(self.features_dim, self.features_dim),
                                      initializer=self.init, trainable=True)
        self.built = True

    # TODO 这里没有去实践Mask和多头的功能，所以比较简单

    def call(self, inputs, **kwargs):
        Q_ = K.dot(inputs, self.WQ)
        K_ = K.dot(self.label_emb, self.WK)

        # step1: 计算S = Q_* K_
        K_ = K.permute_dimensions(K_, [1, 0])
        S_ = K.dot(Q_, K_)
        # step2: 计算类softMax归一化
        A_ = K.softmax(S_, axis=-1)
        if self.is_last_layer is False:
            # step3: 计算 带权V_值
            V_ = K.dot(self.label_emb, self.WV)
            return K.dot(A_, V_)
        else:
            return A_

    def compute_output_shape(self, input_shape):
        if self.is_last_layer is False:
            return input_shape[0], self.steps_dim, self.features_dim
        else:
            return input_shape[0], self.steps_dim, self.categeroy_count


class BertTrainer:
    def __init__(self, folder, seq_len, fine_tune=False):
        self.folder = folder
        self.seq_len = seq_len
        self.fine_tune = fine_tune

    def build(self):
        embed = BERTEmbedding(model_folder=self.folder,
                              task=kashgari.LABELING,
                              trainable=self.fine_tune,
                              sequence_length=self.seq_len)
        model = BiLSTM_CRF_Model(embed)
        return model


# NER的原型网络小样本训练器
class ProtoNetTrainer:
    pass


# 实体关系抽取训练器
class RelationExtractorTrainer:
    pass


# 实体+关系的联合抽取训练器
class JointExtractorTrainer:
    pass
