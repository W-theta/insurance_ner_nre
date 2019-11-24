import math
import numpy as np
from Entity import Document, NamedEntity
from typing import List

CATEGORY = [
    "Amount",  # 用药剂量
    "Anatomy",  # 部位
    "Disease",  # 疾病名称
    "Drug",  # 药品名称
    "Duration",  # 持续时间
    "Frequency",  # 用药频率
    "Level",  # 程度
    "Method",  # 用药方法
    "Operation",  # 手术
    "Reason",  # 病因
    "SideEff",  # 不良反应
    "Symptom",  # 临床表现
    "Test",  # 检查方法
    "Test_Value",  # 检查指标值
    "Treatment"  # 非药方法
]

category2label = dict([(k, v) for v, k in enumerate(CATEGORY)])
label2category = dict([(k, v) for k, v in enumerate(CATEGORY)])


class Sentences:
    def __init__(self, doc_name: str, sents: np.array, labels: np.array, window_size: int, padding_size: int,
                 entities: List[NamedEntity]):
        self.doc_name = doc_name
        self.sents = sents
        self.labels = labels
        self.window_size = window_size
        self.padding_size = padding_size
        self.entities = entities


class SentenceExtractor:
    def __init__(self, window_size: int, padding_size: int, doc: Document):
        self.window_size = int(window_size)
        self.padding_size = int(padding_size)
        self._doc = doc

    def __create_label4doc_seq(self):
        # 根据实体的标注信息建立整篇doc的标注序列
        doc_length = len(self._doc.text)
        label_seq = np.zeros(doc_length)
        entities = self._doc.entities

        for entity in entities:
            start = int(entity.start_pos)
            end = int(entity.end_pos)
            label_seq[start:end] = category2label[entity.category]
        return label_seq

    def extract(self):
        doc_length = len(self._doc.text)
        whole_txt = self._doc.text
        sentence_count = int(math.ceil(doc_length / self.window_size))

        labels4doc = self.__create_label4doc_seq()

        # 切分doc为若干个句子，每个句子长为window_size
        sentences = [whole_txt[i * self.window_size: min(doc_length, (i + 1) * self.window_size)]
                     for i in range(sentence_count)]

        # 切分doc的标注序列，每个子序列长为window_size
        label4sentences = [labels4doc[i * self.window_size: min(doc_length, (i + 1) * self.window_size)]
                           for i in range(sentence_count)]

        s_matrix = None
        l_matrix = None

        for idx, (sentence, labels) in enumerate(zip(sentences, label4sentences)):
            sentence = np.array(sentence)

            if idx == 0:
                # 第一个句子要左端补0，右边部下文
                sentence = np.insert(sentence, 0, np.zeros(self.padding_size))
                sentence = np.append(sentence, sentences[idx + 1][:self.padding_size])

                labels = np.insert(labels, 0, np.zeros(self.padding_size))
                labels = np.append(labels, label4sentences[idx + 1][:self.padding_size])

                s_matrix = sentence
                l_matrix = labels

            elif idx == (sentence_count - 1):
                # 最后一个句子要右端补0
                padding_count = self.window_size + self.padding_size - len(sentence)

                sentence = np.insert(sentence, 0, sentences[idx - 1][-self.padding_size:])
                sentence = np.append(sentence, np.zeros(padding_count))

                labels = np.insert(labels, 0, label4sentences[idx - 1][-self.padding_size:])
                labels = np.append(labels, np.zeros(padding_count))

                s_matrix = np.vstack((s_matrix, sentence))
                l_matrix = np.vstack((l_matrix, labels))

            else:
                # 其余的分别取上下文的padding_size个字符
                sentence = np.insert(sentence, 0, sentences[idx - 1][-self.padding_size:])
                sentence = np.append(sentence, sentences[idx + 1][:self.padding_size])

                labels = np.insert(labels, 0, label4sentences[idx - 1][-self.padding_size:])
                labels = np.append(labels, label4sentences[idx + 1][:self.padding_size])

                if len(sentence) < self.window_size + self.padding_size * 2:
                    # 由于下个句子是最后一句，没有padding_size那么长,所以判定并在右边补0处理
                    padding_count = self.window_size + self.padding_size * 2 - len(sentence)

                    sentence = np.append(sentence, np.zeros(padding_count))
                    labels = np.append(labels, np.zeros(padding_count))

                s_matrix = np.vstack((s_matrix, sentence))
                l_matrix = np.vstack((l_matrix, labels))

        return Sentences(self._doc.doc_name, s_matrix, l_matrix, self.window_size, self.padding_size,
                         self._doc.entities)


def get_sentsArray(sents_set):
    s = None
    l = None
    for idx, sentences in enumerate(sents_set):
        if idx == 0:
            s = sentences.sents
            l = sentences.labels
        else:
            s = np.vstack((s, sentences.sents))
            l = np.vstack((l, sentences.labels))
    return s, l

# if __name__ == "__main__":
#     # 测试切分程序正常
#     root = "data/round1/ruijin_round1_train_20181022/ruijin_round1_train2_20181022/"
#     d = Document("144_15", root)
#     se = SentenceExtractor(50, 10, d).extract()
#     print("")
