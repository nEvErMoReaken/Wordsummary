#-*- coding:utf-8 -*- coding
#@Author:Jiang
#@Date:2020/4/7
#@Time:下午 3:25
#@User:Administrator

#停用词路径
stop_words_path = "E:\WordSummary\data\stop_words.txt"

#原始文章存放路径
origin_data_path = ""

#处理好的用于训练词向量的数据存放路径，格式是每一行一篇文章
text_path = "data/text.txt"


#词向量模型存放路径
w2v_model_path = "E:\WordSummary\model\w2v.model"

#选择训练字向量还是词向量  之所以选择是否训练字向量，是为了防止OOV的问题，
# 训练字向量的话可以避免该问题 ,True 表示训练词向量 ，False 表示训练字向量
# 此处需要注意的是:如果训练字向量 那就不能去停用词！！！！！
use_words_vector = True

#是否去停用词,默认是True表示去停用词
use_stopwords = True

#生成文档向量时，断句用的符号,这个要根据给定的文章的符号格式进行调整 下面是中英文版本的
break_points = [",",".","!","?",";","，","。","！","？","；"]

#提取关键字方法 textRank 和 Tfidf,0 表示textRank 1表示tfidf  本代码暂时只是实现了 textrank算法  如有需要可以在utils里补充
keyword_type = 0

#设置位置权重
locFirst_weight = 1.15
locLast_weight = 1.15

#设定我们想要的摘要句子长度 根据该长度可以求解出长度的权重
summary_len = 10

#每条句子由于长度产生的最小的权重值
minLen_weight = 0.5

#第一次摘要列表句子个数
first_num = 10

#最终我们要得到的摘要句子个数
last_num = 7

#是否保证句子多样性 采用MMR  默认采用该技术
use_MMR = True

#定义MMR算法的alpha值 alpha 值越小 说明需要多样性的程度越大
alpha = 0.5
