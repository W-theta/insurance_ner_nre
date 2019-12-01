from common.Entity import *
from common.Utils import *
from typing import Tuple, Dict
from collections import Counter
import re
import math

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

RELATION = [
    "Test_Disease",  # 检查方法-疾病
    "Symptom_Disease",  # 临床表现-疾病
    "Treatment_Disease",  # 非药治疗-疾病
    "Drug_Disease",  # 药品名称-疾病
    "Anatomy_Disease",  # 部位-疾病
    "Frequency_Drug",  # 用药频率-药品名称
    "Duration_Drug",  # 持续时间-药品名称
    "Amount_Drug",  # 用药剂量-药品名称
    "Method_Drug",  # 用药方法-药品名称
    "SideEff-Drug"  # 不良反应-药品名称
]

category2label = dict([(k, v) for v, k in enumerate(CATEGORY)])
label2category = dict([(k, v) for k, v in enumerate(CATEGORY)])


class DataSet:
    def __init__(self, root_path, file_names, char2idx=None, vocab_size=2000):
        self.root_path = root_path  # type:str
        self.file_names = file_names  # type:List[str]

        self._char2idx = char2idx  # type:Dict[str,int]
        self.vocab_size = vocab_size  # type:int
        self._docs = []  # type:List[Document]

        self.__init_docs(file_names)

    def __getitem__(self, doc_name):
        for d in self._docs:
            if d.doc_name == doc_name:
                return d
            else:
                raise ValueError("不存在doc_name={}的训练文件".format(d.doc_name))

    @property
    def docs(self):
        return self._docs

    @property
    def char2idx(self):
        if self._char2idx is None:
            self._char2idx = self.__init_char_dic()
        return self._char2idx

    @char2idx.setter
    def char2idx(self, new_char2idx: Dict[str, int]):
        self._char2idx = new_char2idx

    def __init_docs(self, file_names):
        docs = []  # type:List[Document]
        for f_name in file_names:
            # 读取text文档
            text = load_file(self.root_path, f_name, "txt")  # type:str
            # 读取entities和ent_pairs
            ann_data = load_file(self.root_path, f_name, "ann")
            entities, entity_pairs = self.__get_entities_and_pairs(ann_data)  # type:NamedEntitySet,List[EntityPair]

            d = Document(f_name, self.root_path, text, entities, entity_pairs)
            docs.append(d)
            self._docs = docs

    @staticmethod
    def __get_entities_and_pairs(ann_data):
        entities = NamedEntitySet()
        ent_pairs = []
        for line in ann_data:
            idx, second_part = line.strip().split("\t", 1)
            if idx[0] == "T":
                label, word = second_part.split("\t")
                category, pos = label.split(' ', 1)
                pos = pos.split(' ')
                obj = NamedEntity(idx, category, int(pos[0]), int(pos[-1]), word)
                entities.add(obj)
            elif idx[0] == "R":
                relation, third_part = second_part.split(" ", 1)
                pattern = re.compile(r"(T\d+)")
                result = pattern.findall(third_part)

                from_ent = entities[str(result[0])]
                to_ent = entities[str(result[1])]

                obj = EntityPair(from_ent, to_ent, relation)
                ent_pairs.append(obj)
        return entities, ent_pairs

    def __init_char_dic(self):
        char_dic = {}
        vocab_size = self.vocab_size
        counter = Counter()
        for doc in self._docs:
            for single_c in doc.text:
                counter[single_c] += 1
        char_dic['_padding'] = 0
        char_dic['_unk'] = 1
        if vocab_size > 0:
            if (vocab_size - 2) <= len(counter):
                most_common_count = vocab_size - 2
            else:
                most_common_count = len(counter)
        else:
            most_common_count = len(counter)
        for idx, (char, _) in enumerate(counter.most_common(most_common_count)):
            char_dic[char] = idx + 2
        return char_dic


class DataProcessor:
    def __init__(self, data_set: DataSet):
        self.data_set = data_set
        self._seqs_4ner_ds = []  # type:List[SeqSet4ner]

    @property
    def seqset_4ner_ds(self):
        return self._seqs_4ner_ds

    def data4NER(self, window=70, pad=10):
        docs = self.data_set.docs

        seq_set_4dataset = []  # type:List[SeqSet4ner]

        for doc in docs:
            whole_txt = doc.text
            doc_len = len(whole_txt)  # type:int

            # vectorize_txt
            char_dic = self.data_set.char2idx  # type:Dict[str,int]
            txt2idx = []
            for c in whole_txt:
                if c in char_dic.keys():
                    txt2idx.append(char_dic[c])
                else:
                    txt2idx.append(char_dic["_unk"])
            whole_txt = txt2idx
            del char_dic

            # vectorize_label
            labels4doc = self.__create_label_embs_4doc(doc)
            seq_count = int(math.ceil(doc_len / window))

            # 切分txt_embs和label_embs，len(seq) = window
            seqs = [whole_txt[i * window: min(doc_len, (i + 1) * window)] for i in range(seq_count)]
            label4seqs = [labels4doc[i * window: min(doc_len, (i + 1) * window)] for i in range(seq_count)]

            s_matrix = None
            l_matrix = None

            seq_set = SeqSet4ner(doc_name=doc.doc_name, window_size=window, padding_size=pad)
            for idx, (s_emb, l_emb) in enumerate(zip(seqs, label4seqs)):
                s_emb = np.array(s_emb)

                if idx == 0:
                    # first_seq:左端补0，右端补下文
                    s_emb = np.insert(s_emb, 0, np.zeros(pad))
                    s_emb = np.append(s_emb, seqs[idx + 1][:pad])

                    l_emb = np.insert(l_emb, 0, np.zeros(pad))
                    l_emb = np.append(l_emb, label4seqs[idx + 1][:pad])

                    s_matrix = s_emb
                    l_matrix = l_emb

                elif idx == (seq_count - 1):
                    # last_seq:左端补上文，右端补0
                    padding_count = window + pad - len(s_emb)

                    s_emb = np.insert(s_emb, 0, seqs[idx - 1][-pad:])
                    s_emb = np.append(s_emb, np.zeros(padding_count))

                    l_emb = np.insert(l_emb, 0, label4seqs[idx - 1][-pad:])
                    l_emb = np.append(l_emb, np.zeros(padding_count))

                    s_matrix = np.vstack((s_matrix, s_emb))
                    l_matrix = np.vstack((l_matrix, l_emb))

                else:
                    # else_seq:左端补上文，右端补下文
                    s_emb = np.insert(s_emb, 0, seqs[idx - 1][-pad:])
                    s_emb = np.append(s_emb, seqs[idx + 1][:pad])

                    l_emb = np.insert(l_emb, 0, label4seqs[idx - 1][-pad:])
                    l_emb = np.append(l_emb, label4seqs[idx + 1][:pad])

                    if len(s_emb) < window + pad * 2:
                        # 由于下文是last_seq，且len(last_seq)<pad,故右端补0对齐
                        padding_count = window + pad * 2 - len(s_emb)

                        s_emb = np.append(s_emb, np.zeros(padding_count))
                        l_emb = np.append(l_emb, np.zeros(padding_count))

                    s_matrix = np.vstack((s_matrix, s_emb))
                    l_matrix = np.vstack((l_matrix, l_emb))

                current_seq = Seq4ner(idx=idx, char_embs=s_emb, label_embs=l_emb)
                seq_set.add(current_seq)

            seq_set_4dataset.append(seq_set)

            self._seqs_4ner_ds = seq_set_4dataset
        return self

    def get_ner_data(self):
        if len(self.seqset_4ner_ds) == 0:
            raise ValueError("未进行数据预处理，请先调用data4NER再执行该接口")
        else:
            s_embs = np.zeros(0)
            l_embs = np.zeros(0)
            for idx, seq_set in enumerate(self.seqset_4ner_ds):
                if idx == 0:
                    s_embs, l_embs = seq_set.get_data()
                else:
                    s_embs = np.vstack((s_embs, seq_set.char_embs))
                    l_embs = np.vstack((l_embs, seq_set.label_embs))
            return s_embs, l_embs

    @staticmethod
    def __create_label_embs_4doc(doc: Document):
        # 根据实体的标注信息建立整篇doc的标注序列
        doc_length = len(doc.text)
        label_seq = np.zeros(doc_length)
        entities = doc.entities

        for entity in entities:
            start = int(entity.start_pos)
            end = int(entity.end_pos)
            label_seq[start:end] = category2label[entity.category]
        return label_seq

    def data4RE(self, position_emb_type="relative_distance"):
        if position_emb_type == "relative_distance":
            seq_emb = np.zeros(0)
            label_emb = np.zeros(0)
            from_pos_emb = np.zeros(0)
            to_pos_emb = np.zeros(0)
            Y = np.zeros(0)
            return seq_emb, label_emb, from_pos_emb, to_pos_emb, Y
        elif position_emb_type == "sin_cos":
            seq_emb = np.zeros(0)
            pos_emb = np.zeros(0)
            X = seq_emb + pos_emb
            Y = np.zeros(0)
            return seq_emb, pos_emb, Y
