import pathlib


def scan_files(path):
    root = pathlib.Path(path)
    # 获取文件名，不包含尾缀
    all_path = [str(path.stem) for path in root.glob('*.txt')]
    return all_path


def load_file(path, file_name, file_type="txt"):
    assert (file_type == "txt") or (file_type == "ann"), \
        "file_type is wrong type. Available type is txt or ann"
    path = path + file_name + "." + file_type

    with open(path, encoding="utf-8") as f:
        if file_type == "txt":
            data = f.read()
        elif file_type == "ann":
            data = f.readlines()
    return data


# TODO 计算F1分数的函数
def f1_score():
    pass
