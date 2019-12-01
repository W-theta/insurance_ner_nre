# 领域知识图谱构建

> 【概述】面向医疗领域的知识图谱构建，主要通过NLP算法提取文本中的领域实体和实体间关系。

## 项目内容

- 通过Brat对语料进行人工标记
- 对命名实体进行识别与提取
  - biLSTM-CRF
  - biLSTM-softmax
  - biLSTM-Attention
- 提取命名实体间的关系
  - PCNN
  - 类DPCNN
- TODO ：基于neo4j的知识图谱构建及应用

## 工程结构

- common    ---通用方法类+实体类
  - Entity.py    ---定义关键实体：包括命名实体、命名实体集、命名实体对、句子序列、句子序列集、文档
  - Utils.py      ---定义通用方法：包括检索语料的文件名、读取加载语料
- Data.py  
  - DataSet     ---定义训练集类
  - DataProcessor      --- 定义训练数据预处理方法，产出可用于训练的数据
- Model.py    ---定义算法模型（包括Word2Vec训练器在内）
- Main.ipynb  ---训练模型的主入口