#-*- coding:utf-8 -*- coding
#@Author:Jiang
#@Date:2020/4/7
#@Time:下午 3:25
#@User:Administrator

"""
实现计算句向量和文档向量的方法
"""
import numpy as np
import jieba
import joblib
import w2v_train
import GlobalParameters
import utils

#定义生成句向量方法
def sentence_vector(sentence,stop_words,model):
    """
    根据词向量模型 和 给定句子 生成句向量
    :param sentence:
    :param stop_words:停用词列表 之所以传入形式  是因为可以再程序启动后 只需要在外部加载一次
    :param model: 词向量模型，之所以通过传参数 是为了项目上线时候 可以再外部加载一次模型，不能在代码内部加载
    :return: 返回句子的向量  是np.arrray() 格式的
    """
    #初始化一个句子向量,维度100 是根据word2vec模型的参数得来的
    vector = np.zeros(100,)

    #判断用的是不是词向量
    if GlobalParameters.use_words_vector:
        #当使用了词向量时候，在判断是否需要去停用词
        if GlobalParameters.use_stopwords:
            #将句子进行分词
            content = jieba.lcut(sentence)
            #计算分词后的长度
            count = 0
            for word in content:
                #假如该词不在停用词表里
                if word not in stop_words:
                    #统计用了几个词语进行向量求和
                    count += 1
                    #将查询后的向量假如进去
                    try:
                      vector += np.array(model[word])
                    #平滑措施，防止oov问题
                    except:
                      pass
            #注意此处 一些句子会出现 ."不！" 这样的话，去停用词后是空列表  导致 count = 0 导致没办法除 所以需要判断count
            if count == 0:
               return np.zeros(100,)
            return vector/count
        #假设不用停用词
        else:
            # 将句子进行分词
            content = jieba.lcut(sentence)
            #计算长度
            length = len(content)
            #遍历该列表
            for word in content:
                vector += np.array(model[word])
            return vector/length
    #假如使用字向量，将不用考虑停用词
    else:
        content = [x for x in sentence]
        #计算长度
        length = len(content)

        #遍历求解
        for word in content:
            vector += np.array(model[word])

        return vector/length

#定义生成文档向量的方法
def doc_vector(text,stop_words,model):
    """
    计算文档向量，句子向量求平均
    :param text: 需要计算的文档
    :param stop_words:停用词表
    :param model:词向量模型
    :return:文档向量
    """

    #获取(位置，句子)列表
    sen_lis = utils.get_sentences(text)
    #提取出句子
    sen_lis = [x[1] for x in sen_lis]
    #定义一个文档初始化向量 100是根据训练的词向量的维度来的
    vector = np.zeros(100,)
    #计算文档里包含多少句子
    length = len(sen_lis)
    #遍历所有句子
    for sentence in sen_lis:
        #获取句向量
        sen_vec = sentence_vector(sentence,stop_words,model)
        #计算文档向量
        vector += sen_vec
        # print(vector)

    #返回文档向量
    return vector/length

if __name__ == "__main__":
    #加载停用词
    stop_words = w2v_train.read_stopwords()

    #加载词向量模型
    model = joblib.load(GlobalParameters.w2v_model_path)
    text = "记得很小的时候，我到楼下去玩，一不小心让碎玻璃割伤了腿，疼得我“哇哇”大哭。爸爸问讯赶来，把我背到了医院，仔仔细细地为我清理伤口《爸爸是医生》、缝合、包扎，妈妈则在一旁流眼泪，一副胆战心惊的样子。我的腿慢慢好了，爸爸妈妈的脸上，才渐渐有了笑容。 一天下午，放学时，忽然下起了倾盆大雨。我站在学校门口，喃喃自语：“我该怎么办？”正在我发愁的时候，爸爸打着伞来了。“儿子，走，回家！”我高兴得喜出望外。这时，爸爸又说话了：“今天的雨太大了，地上到处是水坑，我背你回家！”话音未落，爸爸背起我就走了。一会儿，又听到爸爸说：“把伞往后挪一点，要不挡住我眼了。”我说：“好！”回到家，发现爸爸的衣服全湿透了，接连打了好几个喷嚏。我的眼泪涌出来了。 “可怜天下父母心”，这几年里，妈妈为我洗了多少衣服，爸爸多少次陪我学习玩耍，我已经记不清了。让我看在眼里、记在心里的是妈妈的皱纹、爸爸两鬓的白发。我的每一步成长，都包含了父母太多的辛勤汗水和无限爱心，“可怜天下父母心”！没有人怀疑，父母的爱是最伟大的、最无私的！"
    print(doc_vector(text,stop_words,model))











