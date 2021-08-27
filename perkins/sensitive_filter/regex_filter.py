import re

from perkins.sensitive_filter.params import text_list


def test_regex():
    """
        re.match只匹配字符串的开始，如果字符串开始不符合正则表达式，则匹配失败，函数返回None；而re.search匹配整个字符串，直到找到一个匹配。
    """
    print(re.match('www', 'www.runoob.com').span())  # 在起始位置匹配
    print(re.match('com', 'www.runoob.com'))  # 不在起始位置匹配

    line = "Cats are smarter than dogs"
    matchObj = re.match(r'(.*) are (.*?) .*', line, re.M | re.I)
    if matchObj:
        print("matchObj.group() : ", matchObj.group())
        print("matchObj.group(1) : ", matchObj.group(1))
        print("matchObj.group(2) : ", matchObj.group(2))
    else:
        print("No match!!")

    print(re.search('www', 'www.runoob.com').span())  # 在起始位置匹配
    print(re.search('com', 'www.runoob.com').span())  # 不在起始位置匹配


def test_sub():
    """
        pattern : 正则中的模式字符串。
        repl : 替换的字符串，也可为一个函数。
        string : 要被查找替换的原始字符串。
        count : 模式匹配后替换的最大次数，默认 0 表示替换所有的匹配。
    """
    phone = "2004-959-559 # 这是一个国外电话号码"
    # 删除字符串中的 Python注释
    num = re.sub(r'#.*$', "", phone)
    print("电话号码是: ", num)

    # 删除非数字(-)的字符串
    num = re.sub(r'\D', "", phone)
    print("电话号码是 : ", num)

    # 将匹配的数字乘以 2
    def double(matched):
        value = int(matched.group('value'))
        return str(value * 2)

    s = 'A23G4HFD567'
    print(re.sub('(?P<value>\d+)', double, s))


def test_compile():
    pattern = re.compile(r'\d+', re.I)  # 用于匹配至少一个数字
    m = pattern.match('one12twothree34four')  # 查找头部，没有匹配
    print(m, "\n")
    m = pattern.match('one12twothree34four', 2, 10)  # 从'e'的位置开始匹配，没有匹配
    print(m, "\n")
    m = pattern.match('one12twothree34four', 3, 10)  # 从'1'的位置开始匹配，正好匹配
    print(m, "\n")


def format_name(s):
    s1 = s[0:1].upper() + s[1:].lower();
    return s1;


def test_simple_regex():
    text = "把你的银行卡号发给我"
    re_str = "银行卡|我"
    print(re_str)
    pattern = re.compile(re_str, re.I)
    matchObj = pattern.findall(text)
    print(matchObj)


def test_regex_from_file():
    text = "把你的银行卡号发给我，在本店购买的商铺统一发送银行卡"
    lines = []
    with open("sensitive_words.txt", mode='r', encoding='utf-8') as file:
        lines = file.readlines()
    re_str = ('|'.join(list(map(lambda x: x.strip(), lines))))
    print(re_str)
    pattern = re.compile(re_str, re.I)
    matchObj = pattern.findall(text)
    print(matchObj)


def test_regex_demo02():
    regex_str = r'[给我|发我|发个我|传过来]*.*银行卡.*[给我|发我|发个我|传过来]*'
    pattern = re.compile(regex_str, re.I)
    for item in text_list:
        match_obj = pattern.findall(item)
        print("\n", match_obj)
