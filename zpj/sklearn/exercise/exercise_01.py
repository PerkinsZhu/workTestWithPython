from sklearn import datasets as ds
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
from sklearn.preprocessing import MinMaxScaler

"""
文档地址：
    https://blog.csdn.net/xiaotian127/article/details/86756402
"""


def test_Other():
    iris = ds.load_iris()
    print(iris)
    print("data:\n", iris.data)
    print("target:\n", iris["target"])
    print("target_names:\n", iris["target_names"])
    print("DESCR:\n", iris["DESCR"])


def test_dictVector():
    """
    字典特征抽取
    """
    # sparse =true 则返回的为系数矩阵
    dict = DictVectorizer(sparse=False)
    data = dict.fit_transform([{'city': '北京', 'temperature': 100},
                               {'city': '上海', 'temperature': 60},
                               {'city': '深圳', 'temperature': 30}])
    print("names:\n", dict.get_feature_names())
    print("data:\n", data)


def test_textVector():
    """
    文本特征抽取 这里是按照空格分词的
    """
    # 词频矩阵
    countVector = CountVectorizer()
    data = countVector.fit_transform(['life is short,i like python', 'life is too long,i dislike python'])
    pl(data.toarray(), "data")
    pl(countVector.get_feature_names(), "feature_name")


originText = ['今天很残酷，明天更残酷，后天很美好，但绝大部分是死在明天晚上，所以每个人不要放弃今天。',
              '我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。',
              '如果只用一种事物了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。']


def test_chinese_textVector():
    """
    中文 文本特征抽取，需要进行分词
    """
    # 词频矩阵
    countVector = CountVectorizer()
    newTextArray = cut_word(originText)
    pl(newTextArray)
    data = countVector.fit_transform(newTextArray)
    pl(data.toarray(), "data")
    pl(countVector.get_feature_names(), "feature_name")


def test_TFIDF():
    """
    提取特征词，关键词
    主要思想：如果一个词或短语在一篇文章中出现的概率高，并且在其他文章中很少出现，则认为此词或短语具有很好的类别区分能力，适合用来分类
    文本分类的另一种方式：
    tf idf(
        tf：term frequency;
        idf: 逆文档频率inverse document frequency log(总文档数量/该词出现的文档数量)
        )
    tf * idf为这个词在文章中的重要程度
    """
    tf = TfidfVectorizer()
    newTextArray = cut_word(originText)
    data = tf.fit_transform(newTextArray)
    pl(data)
    pl(data.toarray)
    pl(tf.get_feature_names())


def cut_word(text_array):
    result = []
    for item in text_array:
        str = ' '.join(list(jieba.cut(item)))
        result.append(str)
    return result


def pl(text, tip=""):
    print(tip + "\n", text)


def test_mm():
    """
    归一化处理
    x'=(x-min)/(max-min), x''=x'*(mx-mi)+mi
    """
    # 缩放到 [2,5]之间
    mm = MinMaxScaler(feature_range=(2, 5))
    data = mm.fit_transform([[90, 2, 10, 40], [60, 4, 15, 45], [75, 3, 13, 46]])
    pl(data)
