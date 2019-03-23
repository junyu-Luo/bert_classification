import pandas as pd
from tqdm import trange
import pickle
import os
from collections import Counter


def save_pkl(filename, content):
    file = open('./pkl/' + filename, 'wb')
    pickle.dump(content, file)
    file.close()


def load_pkl(filename):
    with open('./pkl/' + filename, 'rb') as file:
        return pickle.load(file)


NAME = []
USE_NAME = []
CODE = []
data = pd.read_excel('./data/code2name.xlsx')
result = []
for i in trange(len(data)):
    NAME.append(data['通用名'][i])
    USE_NAME.append(data['给药途径'][i])
    CODE.append(data['药品代码'][i])
assert len(NAME) == len(CODE) == len(USE_NAME)

code2name = {}
code2use_name = {}
for i in trange(len(NAME)):
    code2name[CODE[i]] = NAME[i]
    code2use_name[CODE[i]] = USE_NAME[i]


save_pkl('code2name.pkl',code2name)
save_pkl('code2use_name.pkl',code2use_name)


# df = pd.DataFrame({'text':text,'result':result})
# df.to_excel("to_xmk.xlsx")