from itertools import groupby
from Data_Set import DataSet
from Prepare_sents import Sentences, category2label
from common.Entity import Document, NamedEntity
import numpy as np
from typing import List


def merge_preds(testset: DataSet, preds: np.array, window_size: int, padding_size: int) -> List[Document]:
    """
    Arg:将含有padding的多条预测片段序列，合并成1个完整的文章预测序列
    """
    preds = np.argmax(preds, axis=-1)
    sents_set = testset.get_sents_set(window_size, padding_size)  # type:List[Sentences]
    docs = testset.docs  # type:List[Document]

    pos_flag = 0
    pre_docs = []
    for sentences, doc in zip(sents_set, docs):
        # 每一个doc的行数
        sents_lines = sentences.sents.shape[0]
        doc_labels = np.zeros(0)
        start_pos = pos_flag
        end_pos = pos_flag + sents_lines
        for pred_line in preds[start_pos:end_pos]:
            doc_labels = np.append(doc_labels, pred_line[padding_size:(padding_size + window_size)])
        d = Document(sentences.doc_name, "", doc.text, __labels2entities(doc_labels))
        pos_flag = end_pos
        pre_docs.append(d)
    return pre_docs


def __labels2entities(doc_labels: np.array) -> List[NamedEntity]:
    pos_flag = 0
    entities = []
    for idx, (category, group) in enumerate(groupby(doc_labels)):
        start = pos_flag
        end = pos_flag + len(list(group))
        if category != 0:
            entity = NamedEntity(idx, category, start, end, "")
            entities.append(entity)
        pos_flag = end
    return entities


def f1_score(pre_docs: List[Document], source_docs: List[Document], style='all'):
    """
    Arg:计算F1分数
    """
    pre_entities_count = 0
    source_entities_count = 0
    right_entities_count = 0

    for pre_doc, source_doc in zip(pre_docs, source_docs):
        pre_entities = pre_doc.entities
        source_entities = source_doc.entities

        pre_entities_count += len(pre_entities)
        source_entities_count += len(source_entities)
        right_entities_count += __count_intersects(pre_entities, source_entities,style)

    p = right_entities_count / pre_entities_count
    r = right_entities_count / source_entities_count
    f1 = 2 * p * r / (p + r)
    return f1, p, r


def __count_intersects(pred_ent_list: List[NamedEntity], source_ent_list: List[NamedEntity], style='all') -> int:
    num_hits = 0
    source_ent_list = source_ent_list.copy()
    for ent_a in pred_ent_list:
        hit_ent = None
        for ent_b in source_ent_list:
            if style == 'all':
                if __check_match_all(ent_a, ent_b):
                    hit_ent = ent_b
                    break
            else:
                if __check_match(ent_a, ent_b):
                    hit_ent = ent_b
                    break
        if hit_ent is not None:
            num_hits += 1
            source_ent_list.remove(hit_ent)
    return num_hits


def __check_match(ent_a: NamedEntity, ent_b: NamedEntity) -> bool:
    return (ent_a.category == category2label[ent_b.category] and
            max(int(ent_a.start_pos), int(ent_b.start_pos)) < min(int(ent_a.end_pos), int(ent_b.end_pos)))


def __check_match_all(ent_a: NamedEntity, ent_b: NamedEntity) -> bool:
    return ((ent_a.category == category2label[ent_b.category]) and
            (int(ent_a.start_pos) == int(ent_b.start_pos)) and
            (int(ent_a.end_pos), int(ent_b.end_pos)))
