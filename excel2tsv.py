import pandas as pd
from tqdm import trange
import pickle
import os
from collections import Counter

def write_file(filename,str):
    """
    写入文件
    :param str: 字符串
    :return: 无
    """
    writefile = open("./data/"+filename, 'a+',encoding='utf-8')
    writefile.write(str + '\n')
    writefile.close()


def save_pkl(filename, content):
    file = open('./pkl_save/' + filename, 'wb')
    pickle.dump(content, file)
    file.close()


def load_pkl(filename):
    with open('./pkl_save/' + filename, 'rb') as file:
        return pickle.load(file)


def clean_data(string):
    if type(string) != float:
        string = string.replace("　","").replace(" ","").replace("&","").replace("#","").replace("@","").replace("▲","").replace("◤","").strip()
    else:
        string = ''
    return string



NAME = []
SPECS = []
USE_NAME = []
CODE = []
print("读取文件中")
data = pd.read_excel('./data/train.xlsx')
print("读取数据中")
for i in trange(len(data)):
    NAME.append(clean_data(data['NAME'][i]))
    SPECS.append(clean_data(data['SPECS'][i]))
    USE_NAME.append(clean_data(data['USE_NAME'][i]))
    CODE.append(clean_data(data['CODE'][i]))
assert len(NAME) == len(SPECS) == len(USE_NAME) == len(CODE)
# print(len(NAME),NAME)
# print(len(SPECS),SPECS)
# print(len(USE_NAME),USE_NAME)
# print(len(CODE),CODE)
for i in trange(len(NAME)):
    write_file('train.tsv',CODE[i]+'    '+NAME[i]+'。'+SPECS[i]+'。'+USE_NAME[i]+'。')






