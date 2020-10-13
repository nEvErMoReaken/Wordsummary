#-*- coding:utf-8 -*- coding
#@Author:Jiang
#@Date:2020/4/7
#@Time:下午 3:25
#@User:Administrator

import utils
import numpy as np
import GlobalParameters
import generate_vector
import w2v_train
import joblib
from gensim.models import Word2Vec

#根据相似度 关键词权重 以及 句子长度权重 先粗略计算出一个摘要句子列表
def get_first_summaries(text,stopwords,model):
    """

    :param text: 文档
    :param stopwords: 停用词
    :param model: 词向量模型
    :return: 摘要列表  按照权重从大到小排列[(句子，权重),(句子，权重)]
    """
    #获取（位置，句子）列表
    sentences = utils.get_sentences(text)

    #获取句子列表
    sen_lis = [x[1] for x in sentences]
    # print(sen_lis)
    #获取文档向量
    docvec = generate_vector.doc_vector(text,stopwords,model)

    #获取句子向量列表
    sen_vecs = []
    for i in range(len(sen_lis)):
        #假设是首句
        if i == 0 :
            sen_vecs.append(generate_vector.sentence_vector(sen_lis[i], stopwords, model)*GlobalParameters.locFirst_weight)
        #如果是最后一句
        elif i == len(sen_lis)-1:
            sen_vecs.append(generate_vector.sentence_vector(sen_lis[i], stopwords, model) * GlobalParameters.locLast_weight)
        #如果是中间的句子
        else:
            sen_vecs.append(generate_vector.sentence_vector(sen_lis[i], stopwords, model))

    #计算余弦值列表
    cos_lis = [utils.cos_dist(docvec,x) for x in sen_vecs]

    #计算关键词权重列表
    #获取关键词
    keywords = utils.get_keywords(text)

    #计算权重
    keyweights = [utils.keyword_weight(x,keywords) for x in sen_lis]

    #计算长度权重
    len_weigths = [utils.len_weight(x) for x in sen_lis]

    #根据余弦相似度 关键词权重 长度权重 计算每个句子最终权重
    final_weights = [cos*keyword*length for cos in cos_lis for keyword in keyweights for length in len_weigths]

    #形成最后的（句子，权重列表）
    final_lis = []
    for sen,weight in zip(sen_lis,final_weights):
        final_lis.append((sen,weight))

    #将句子按照权重大小 从高到低排序
    final_lis = sorted(final_lis,key=lambda x:x[1],reverse=True)

    #取出第一次摘要的橘子个数
    final_lis = final_lis[:GlobalParameters.first_num]

    return final_lis

#定义MMR算法 保证摘要多样性
def MMR(final_lis,stopwords,model):
    """
    根据MMR算法 保证摘要句子多样性
    :param final_lis: 初步摘要（句子，权重）列表
    :param stopwords:停用词表
    :param model:词向量模型
    :return:最终摘要的句子列表
    """
    #根据final_lis 获取句子列表
    sen_lis = [x[0] for x in final_lis]
    #权重列表
    weight_lis = [x[1] for x in final_lis]
    #定义摘要列表
    summary_lis = []
    #首先挑出来权重最大的一句话，它必然是摘要列表中的一句
    summary_lis.append(sen_lis[0])
    #为了方便处理 将作为最终摘要的句子 从预摘要列表里删除掉
    del sen_lis[0]
    del weight_lis[0]

    #根据要求个数摘要句子
    #如果只摘要一个句子 直接将结果返回就可以了
    if GlobalParameters.last_num == 1:
        return summary_lis
    #摘要不止一个句子 需要进行计算求解
    else:
        for i in range(len(sen_lis)):
            # 所有候选句子的向量列表
            vec_lis = [generate_vector.sentence_vector(x, stopwords, model) for x in sen_lis]
            #已经作为摘要的句子向量列表
            summary_vec =  [generate_vector.sentence_vector(x, stopwords, model) for x in summary_lis]
            #定义各个句子的得分情况
            scores = []

            for vec1 in vec_lis:
                #计数器
                count = 0
                #初始化句子分数
                score = 0
                for vec2 in summary_vec:
                    score +=GlobalParameters.alpha*weight_lis[count]-(1-GlobalParameters.alpha)*utils.cos_dist(vec1,vec2)
                #求新句子与最终摘要句子的平均相似度
                count += 1
                scores.append(score/len(summary_vec))

            #根据最大分数的下标  求解对应的句子加入到摘要里面 通过array求解
            scores = np.array(scores)
            index = np.argmax(scores)
            #将对应句子加入到摘要列表
            summary_lis.append(sen_lis[index])

            #将对应句子 和 权重从预摘要列表里删除
            del sen_lis[index]
            del weight_lis[index]
        #返回指定需要的句子个数
        return summary_lis[:GlobalParameters.last_num]
#获取最终摘要
def get_last_summaries(text,final_lis,stopwords,model):
    """
    获取最终的摘要列表
    :param stopwords: 停用词
    :param model: 词向量模型
    :return: 摘要列表[摘要1，摘要2...]
    """
    #判断是否用MMR
    if GlobalParameters.use_MMR:
        results =  MMR(final_lis,stopwords,model)
    else:
        results =  final_lis[:GlobalParameters.last_num]
        #注意此处的results 是以元祖为元素的列表 需要将句子取出来
        results = [x[0] for x in results]
    #为了使得句子读起来连贯 我们按照摘要句子在原始文章里的位置信息 进行排序
    sentences = utils.get_sentences(text)  #[(1,句子1),(2,句子2)。。]
    # print("句子是",sentences)
    #定义摘要列表 [（句子，位置）,（句子，位置）..]
    summaries = []

    for summary in results:
        for sentence in sentences:
            if summary == sentence[1]:
                summaries.append((summary,sentence[0]))
    # print("summaries:",summaries)
    #根据位置排序
    summaries = sorted(summaries,key=lambda x:x[1])

    #获取最终摘要句子 不要位置信息
    summaries = [x[0] for x in summaries]

    return summaries
def get_summary(content):
    # 加载停用词
    stopwords = w2v_train.read_stopwords ()
    # 加载模型
    model = joblib.load ( GlobalParameters.w2v_model_path )
    #获取用户数据
    # content = input()
    #获取初次摘要列表
    final_lis = get_first_summaries(content,stopwords,model)
    #获取最终摘要列表
    summaries = get_last_summaries(content,final_lis,stopwords,model)
    #将获得摘要拼接
    summary = ",".join(summaries)
    return summary

if __name__ == "__main__":
    # 加载停用词
    stopwords = w2v_train.read_stopwords()
    #加载模型
    model = joblib.load(GlobalParameters.w2v_model_path)
    text="北京电视台文艺频道召开暑期改版发布会，宣布推出由徐春妮主持的全新节目《春妮的周末时光》。春妮称新栏目形式独特，把家搬进了录影棚，明星以朋友做客的形式出现，还大秀他们拿手的家务活。全景访谈栏目《春妮的周末时光》，是ＢＴＶ文艺频道首次以主持人为中心打造的节目。它摒弃了常规电视节目的舞美设计，将整个舞台设计成春妮家的客厅，明星嘉宾褪去光环，融入在家庭的环境中，还原最真实的一面。主持人徐春妮透露了节目前几期的内容：第一期，李莉、韩红、张绍刚、撒贝宁将揭露闺密损友那点事，好友四人相互爆料生活中与屏幕形象极为反差的一面，并就张绍刚事件共同探讨当下年轻人的择业观。韩红还将在节目现场泡制功夫茶，尽显温柔女人味。第二期，成龙将携新七小福讲述彼此的师徒情、兄弟情、父子情，展现与电影角色中不一样的侠骨柔情。第三期，冯远征、梁丹妮、濮存昕将畅谈人艺６０年，展现戏里戏外不同角色、多彩的人生故事。节目将于７月７日起，在每周六１９：３５播出。"
    final_lis = get_first_summaries(text,stopwords,model)
    print(final_lis)
    #生成最终摘要
    print(get_last_summaries(text,final_lis,stopwords,model))







