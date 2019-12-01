from typing import List, Dict
import numpy as np


class NamedEntity:
    def __init__(self, idx: str, category: int, start_pos: int, end_pos: int, name: str):
        self.idx = idx
        self.category = category
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.name = name

    def reset_pos(self, moving_step: int):
        new_start = self.start_pos + moving_step
        new_end = self.end_pos + moving_step
        return NamedEntity(self.idx, self.category, new_start, new_end, self.name)

    def __lt__(self, other):
        return self.start_pos < other.start_pos

    def __eq__(self, other):
        if (self.category == other.category) & \
                (self.start_pos == other.start_pos) & \
                (self.end_pos == other.end_pos):
            return True
        else:
            return False


class NamedEntitySet:
    def __init__(self, ents: List[NamedEntity] = None):
        if ents is None:
            self._ent_dic = {}
        else:
            self._ent_dic = dict([(ent.idx, ent) for ent in ents])

    def add(self, ent: NamedEntity):
        idx = ent.idx
        if idx not in self._ent_dic.keys():
            self._ent_dic[idx] = ent

    def get_all(self) -> List[NamedEntity]:
        return [self._ent_dic[key] for key in self._ent_dic.keys()]

    def __getitem__(self, idx):
        if isinstance(idx, str):
            if idx in self._ent_dic.keys():
                return self._ent_dic[idx]
            else:
                raise ValueError("idx = {}不存在。".format(idx))


class EntityPair:
    def __init__(self, from_ent: NamedEntity, end_ent: NamedEntity, relation_type: str):
        self.from_ent = from_ent
        self.end_ent = end_ent
        self.relation_type = relation_type


class OriginSentence:
    def __init__(self, doc_name: str, sents_text: str, from_pos: int, end_pos: int,
                 ent_set: NamedEntitySet, ent_pairs: List[EntityPair] = None):
        self.doc_name = doc_name
        self.sents_text = sents_text
        self.from_pos = from_pos
        self.end_pos = end_pos
        self.ent_set = ent_set

        if ent_pairs is None:
            pass
            # TODO 根据pair的规则，创建pairs
        else:
            self.ent_pairs = ent_pairs


class Seq4ner:
    def __init__(self, idx: int, char_embs: np.ndarray, label_embs: np.ndarray):
        self.idx = idx
        self.char_embs = char_embs
        self.label_embs = label_embs


class SeqSet4ner:
    def __init__(self, doc_name: str, window_size: int, padding_size: int,
                 char_embs: np.ndarray = np.zeros(0), label_embs: np.ndarray = np.zeros(0),
                 seqs_4_ner: Dict[int, Seq4ner] = None):
        self.doc_name = doc_name
        self._char_embs = char_embs
        self._label_embs = label_embs
        self.window_size = window_size
        self.padding_size = padding_size
        if seqs_4_ner is None:
            self._seqs_4_ner = {}  # type:Dict[int,Seq4ner]
        else:
            self._seqs_4_ner = seqs_4_ner

    @property
    def char_embs(self):
        return self._char_embs

    @char_embs.setter
    def char_embs(self, embs):
        self._char_embs = embs

    @property
    def label_embs(self):
        return self._label_embs

    @label_embs.setter
    def label_embs(self, embs):
        self._label_embs = embs

    @property
    def seqs_dict(self):
        return self._seqs_4_ner

    def add(self, seq: Seq4ner):
        if seq.idx not in self._seqs_4_ner.keys():
            self._seqs_4_ner[seq.idx] = seq
        else:
            raise ValueError("在seqs_4_ner已存在idx为{}的seq。".format(seq.idx))

    def get_data(self):
        return self._char_embs, self._label_embs


class Seq4re:
    def __init__(self, doc_name: str, seq_text: str,
                 max_seq_len: int,
                 start_pos, end_pos,
                 ent_pair: List[EntityPair],
                 position_embedding_type="relative_distance",  # 或者是“sin_cos”
                 sents_num_per_seq=2, stride_size=1):
        self.doc_name = doc_name
        self.seq_text = seq_text
        if max_seq_len < 0:
            raise ValueError("max_seq_len小于等于0。请修改参数值保证参数不为0")
        self.max_seq_len = max_seq_len
        self.start_pos = start_pos
        self.ent_pos = end_pos
        self.ent_pair = ent_pair
        self.position_embedding_type = position_embedding_type
        self.sents_num_per_seq = sents_num_per_seq
        self.stride_size = stride_size

    def get_ent_pairs(self):
        pass

    def __position_embedding(self):
        pass


class SeqSet4re:
    def __init__(self):
        pass


class Document:
    def __init__(self, doc_name: str, root_path: str, text: str, entities: NamedEntitySet,
                 entity_pairs: EntityPair = None):
        self.doc_name = doc_name
        self._root_path = root_path
        self._text = text
        self._entities = entities
        self._entity_pairs = entity_pairs

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, new_text):
        self._text = new_text

    @property  # 不定义setter就变成了一个只读属性
    def entities(self):
        return self._entities

    @property
    def entity_pairs(self):
        return self._entity_pairs
