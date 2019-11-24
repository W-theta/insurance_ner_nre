from Utils import load_file


class NamedEntity:
    def __init__(self, idx, category, start_pos, end_pos, name):
        self.idx = idx
        self.category = category
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.name = name

    def reset_pos(self, moving_step):
        new_start = self.start_pos + moving_step
        new_end = self.end_pos + moving_step
        return NamedEntity(self.idx, self.category, new_start, new_end, self.name)

    def __lt__(self, other):
        return self.start_pos < other.start_pos

    # def __repr__(self):
    #     return repr("{}_{}_[{}-{}]_[{}]".format(self.idx, self.category, self.start_pos, self.end_pos, self.name))


class Document:
    def __init__(self, doc_name, root_path):
        self.doc_name = doc_name

        self._root_path = root_path
        self._text = load_file(self._root_path, self.doc_name, file_type="txt")

        ann_lines = load_file(self._root_path, self.doc_name, file_type="ann")
        named_entities = []
        for line in ann_lines:
            entity = self.__create_named_entity(line)
            named_entities.append(entity)

        self._entities = sorted(named_entities, key=lambda x: x.start_pos)

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, new_text):
        self._text = new_text

    @property  # 不定义setter就变成了一个只读属性
    def entities(self):
        return self._entities

    @staticmethod
    def __create_named_entity(line):
        idx, label, word = line.strip().split("\t")
        category, pos = label.split(' ', 1)
        pos = pos.split(' ')
        return NamedEntity(idx, category, pos[0], pos[-1], word)