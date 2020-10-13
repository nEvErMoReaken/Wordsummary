#-*- coding:utf-8 -*- coding
#@Author:Jiang
#@Date:2020/4/7
#@Time:下午 3:25
#@User:Administrator

import numpy as np
import GlobalParameters
import re
from  jieba import analyse

#定义获取文档断句列表 和 位置信息方法
def get_sentences(text):
    """
    将文档切割成句子
    :param text: 需要进行断句的文档
    :return: 返回一个列表，包含句子信息和位置信息[(1,句子1),(2,句子2),()...(-1,juzi..)]
    """
    # 读取断句符号
    break_points = GlobalParameters.break_points
    # 定义盛放文档的断句
    sen_lis = []

    # 先将text中的所有断句符号替换成 "."
    for point in break_points:
        text = text.replace(point, ".")

    # 根据"."进行断句操作
    sen_lis = text.split(".")
    # 去掉断句后的所有空字符串
    sen_lis = [x for x in sen_lis if x != ""]
    #将位置信息和句子信息封装在一个列表里
    results = []
    for i in range(len(sen_lis)):
        if i != len(sen_lis)-1:
            results.append((i+1,sen_lis[i]))
        #最后一句话的时候 位置是-1
        else:
            results.append((-1,sen_lis[i]))

    return results


#定义余弦函数
def cos_dist(vec1, vec2):
    """
    :param vec1: 向量1
    :param vec2: 向量2
    :return: 返回两个向量的余弦相似度值
    """
    dist1 = float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
    return dist1


#textRank求解关键字
def get_keywords(text):
    """
    返回关键字列表
    :param text: 需要提取关键字的文档
    :return: 返返回关键字列表
    """
    #假设是采用textRank算法
    if GlobalParameters.keyword_type == 0:
        textrank = analyse.textrank
        keywords = textrank(text)
        print(keywords)
        return keywords

    #采用tfidf求解
    else:
        pass


#定义求解句子含有关键字个数的权重值
def keyword_weight(sentence,keywords):
    """
    获取一个句子在一篇文档里的关键字权重值
    :param sentence: 对应句子
    :param keywords: 关键词列表
    :return: 一个float类型的数字
    """
    #计算关键字个数
    count = 0
    for keyword in keywords:
        count += sentence.count(keyword)
    #如果一个句子中不包含关键字 那么权重就是0
    return count/len(keywords)


#定义句子长度的权重
def len_weight(sentence):
    """
    计算句子长度
    :param sentence: 求解的句子
    :return: 句子长度权重值
    """
    #假设求解句子长度小于我们想要的句子长度
    if len(sentence)<=GlobalParameters.summary_len:
        if len(sentence)/GlobalParameters.summary_len>GlobalParameters.minLen_weight:
            return len(sentence)/GlobalParameters.summary_len
        else:
            return GlobalParameters.minLen_weight
    #假设长度大于我们想要的摘要长度
    else:
        #注意 此时如果句子长度大于我们想要摘要的句子长度两倍 那他的权重就是负数 基本不会被选取为摘要
        # return 1-(len(sentence)-GlobalParameters.summary_len)/GlobalParameters.summary_len
        #我们采用下面这个策略 起码保证句子的长度权重值不会为负数  最低为0.5
        if 1-(len(sentence)-GlobalParameters.summary_len)/GlobalParameters.summary_len>GlobalParameters.minLen_weight:
            return 1-(len(sentence)-GlobalParameters.summary_len)/GlobalParameters.summary_len
        else:
            #如果小于阈值 那么就返回最小长度权重值
            return GlobalParameters.minLen_weight
















