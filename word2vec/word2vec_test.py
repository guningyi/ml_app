#!/usr/bin/env python
# -*- coding:utf-8 -*-
from gensim.models import word2vec
import logging
import sys
reload(sys)
sys.setdefaultencoding('utf8')

# http://blog.csdn.net/u013378306/article/details/54629935?ABstrategy=codes_snippets_optimize_v3
if __name__ == '__main__':
    pass
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.Text8Corpus(u'C:/Users/100plus/Desktop/word.txt')  # 加载语料
    #sentences = word2vec.Text8Corpus(u'C:/Users/eniiguu/Desktop/word.txt')  # 加载语料
    model = word2vec.Word2Vec(sentences, size=200)  # 训练skip-gram模型; 默认window=5

    # 计算两个词的相似度/相关程度
    y1 = model.similarity(u'大概', u'陆尘')
    print u'【大概】和【难道】的相似度为：', y1
    print '--------\n'

    # 计算某个词的相关词列表
    y2 = model.most_similar(u'同样', topn=20)  # 20个最相关的
    print u'和【同样】最相关的词有：\n'
    for item in y2:
        print item[0], item[1]
    print '--------\n'

    # 寻找对应关系
    print u'书-不错，质量-'
    y3 = model.most_similar([u'质量', u'不错'], [u'书'], topn=3)
    for item in y3:
        print item[0], item[1]
    print '--------\n'

    # 寻找不合群的词
    y4 = model.doesnt_match(u"书 书籍 教材 很".split())
    print u'不合群的词：', y4
    print '--------\n'

    # 保存模型，以便重用,保存后会生成一个文件，里面存的是语料库的向量
    model.save(u'书评.model')
    # 对应的加载方式
    # model_2 = word2vec.Word2Vec.load("text8.model")

    # 以一种C语言可以解析的形式存储词向量
    model.save_word2vec_format(u'书评.model.bin', binary=True)
    # 对应的加载方式
    # model_3 = word2vec.Word2Vec.load_word2vec_format("text8.model.bin", binary=True)