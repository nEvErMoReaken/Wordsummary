
import GlobalParameters
import jieba
import joblib#用于保存训练好的词向量模型
from gensim.models import Word2Vec
import os

#定义读取停用词的方法
def read_stopwords():
    """
    读取停用词表
    :return: 返回停用词列表[".","的",...]
    """
    #定义盛放停用词的列表
    stop_words = []
    with open(GlobalParameters.stop_words_path,"r",encoding="utf8") as f:
        for line in f:
            #判断读取的是否为空
            if line:
                #将读取的停用词去除空格后添加到停用词列表里
                stop_words.append(line.strip())
    #print(stop_words)
    return stop_words

#读取原始数据
def get_sentences():
    """
    读取原始数据，并将数据处理成word2vec模型需要的格式
    :return: 返回模型训练需要的列表格式[["","",..],["","",],[],..]
    """
    #判断是否需要去停用词
    if GlobalParameters.use_stopwords:
        stop_words = read_stopwords()

    #定义盛放数据的列表
    sentences = []

    #判断是否训练词向量
    if GlobalParameters.use_words_vector:
        with open(GlobalParameters.text_path,"r",encoding="utf8") as f:
            for line in f:
                #判断读取是否为空
                if line:
                    #去除文章前后空格
                    content = line.strip()
                    #进行分词操作
                    content = jieba.lcut(content)
                    #判断是否去停用词
                    if GlobalParameters.use_stopwords:
                        for word in content:
                            if word in stop_words:
                                content.remove(word)
                    #如果最终内容不为空 就加入到sentences
                    if content:
                        sentences.append(content)
    #如果用字向量,不用jieba 分词,挨个加到句子里 注意 字向量肯定不用去停用词
    else:
        with open(GlobalParameters.text_path,"r",encoding="utf8") as f:
            for line in f:
                #判断读取是否为空
                if line:
                    #去除文章前后空格
                    content = line.strip()
                    content = [x for x in content]
                    #判断 content 是否为空
                    if content:
                        sentences.append(content)
    return sentences

#训练向量模型
def train():
    """
    训练好模型 并 保存到指定位置
    :return:
    """
    #读取数据
    sentences = get_sentences()

    #训练模型
    model = Word2Vec(sentences,size=100,window=3,min_count=1,iter=1)

    #保存模型
    joblib.dump(model,GlobalParameters.w2v_model_path)


if __name__ == "__main__":
    #如果模型不存在就开始训练

    if not os.path.exists(GlobalParameters.w2v_model_path):
        train()

    #加载模型
    model = joblib.load(GlobalParameters.w2v_model_path)

    #查询词向量
    print(model)




