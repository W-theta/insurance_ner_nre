from common.Entity import Document, NamedEntity
from common.Utils import scan_files
from typing import List


class build_dataSet:
    def __init__(self, root_path: str, sents_per_seq: int, max_seq_len: int):
        self.root_path = root_path
        self.sents_per_seq = sents_per_seq
        self.max_seq_len = max_seq_len

        docs = []
        for f in scan_files(self.root_path):
            d = Document(f, self.root_path)
            docs.append(d)

        self._docs = docs  # type:List[Document]
        print(len(docs))

    # TODO 以n句话作为一个训练样本
    def seqs(self):
        doc_seq_list = []
        for doc in self._docs:
            doc_text = doc.text  # type:str
            # 以文章中的句号切分为自然句
            sents = doc_text.split("。")
            sents_per_seq = self.sents_per_seq

            seq_list = []  # type:List[str]
            current_seq = ""  # type:str
            for idx, sent in enumerate(sents):
                if idx == 0:
                    current_seq = sent
                    current_seq += "。"
                elif idx != 0:
                    if idx % sents_per_seq != 0:
                        current_seq += sent
                        current_seq += "。"
                        if idx == len(sent) - 1:
                            seq_list.append(current_seq)
                    elif idx % sents_per_seq == 0:
                        seq_list.append(current_seq)
                        current_seq = sent
                        current_seq += "。"
                        if idx == len(sent) - 1:
                            seq_list.append(current_seq)

            doc_seq_list.append(seq_list)
        return doc_seq_list

    def split_original_sents(self, doc: Document):
        doc_text = doc.text
        sents = doc_text.split("。")

        pos_flag = 0
        orig_sents = []  # type:List[Original_sents]
        for idx, sent in enumerate(sents):
            sent += "。"
            doc_name = doc.doc_name

            start = pos_flag
            end = pos_flag + len(sent)

            o = Original_sents(doc_name, sent, start, end)
            orig_sents.append(o)

            pos_flag = end
        return orig_sents


class Original_sents:
    def __init__(self, doc_name, sents_text, from_pos, end_pos):
        self.doc_name = doc_name
        self.sents_text = sents_text
        self.from_pos = from_pos
        self.end_pos = end_pos


class entity_pair:
    def __init__(self, from_ent: NamedEntity, to_ent: NamedEntity,
                 from_ent_start: int, from_ent_end: int,
                 to_ent_start: int, to_ent_end: int):
        self.from_ent = from_ent
        self.to_ent = to_ent
        self.from_ent_start = from_ent_start
        self.from_ent_end = from_ent_end
        self.to_ent_start = to_ent_start
        self.to_ent_end = to_ent_end


if __name__ == "__main__":
    root = "../data/round1/train/"
    tt = build_dataSet(root_path=root, sents_per_seq=2, max_seq_len=1000)
    a = tt.seqs()
    print("")
