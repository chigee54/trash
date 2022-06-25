# 创建语料，清洗数据
import os
import opencc
import jieba
import re
from tqdm import tqdm

user_dict_path = '/content/drive/Shareddrives/zhangzhijie5454@gmail.com/file/wordsim297_dict.txt'
jieba.load_userdict(user_dict_path)
converter = opencc.OpenCC('t2s.json')

def remove_empty_paired_punc(in_str):
  return in_str.replace('（', '').replace('《', '').replace('】', '').replace('【','').replace('[','').replace(']', '').replace('》', '').replace('）','')

def remove_html_tags(in_str):
  html_pattern = re.compile(r'<[^>]+>', re.S)
  return html_pattern.sub('', in_str)

def remove_control_chars(in_str):
  control_chars = ''.join(map(chr, list(range(0, 32)) + list(range(127, 160))))
  control_chars = re.compile('[%s]' % re.escape(control_chars))
  return control_chars.sub('', in_str)


# 读取训练数据
file_list = os.listdir('/content/drive/Shareddrives/zhangzhijie5454@gmail.com/text')
file_list = [f'/content/drive/Shareddrives/zhangzhijie5454@gmail.com/text/{x}' for x in file_list]
# 分词
for file in tqdm(file_list):
  sub_file_list = os.listdir(file)
  sub_file_list = [file+f'/{x}' for x in sub_file_list]
  for sub_file in sub_file_list:
    with open(sub_file, 'r', encoding='utf-8') as f:
      for line in f.readlines():
        line = line.strip()
        line2 = converter.convert(line)
        
        line3 = remove_empty_paired_punc(line2)
        line3 = remove_html_tags(line3)
        line3 = remove_control_chars(line3)

        line_cut = jieba.cut(line3)  # 使用jieba进行分词
        
        result = ' '.join(line_cut)  # 把分词结果用空格组成字符串
        
        with open('/content/drive/Shareddrives/zhangzhijie5454@gmail.com/file/clean.txt', 'a', encoding='utf-8') as fw:
          fw.write(result)  # 把分好的词写入到新的文件里面
          pass
        pass
      pass
    pass
  pass


# 开始训练模型并保存
from gensim.models import word2vec
from tqdm import tqdm
sentences = word2vec.LineSentence('/content/drive/Shareddrives/zhangzhijie5454@gmail.com/file/clean.txt')
model = word2vec.Word2Vec(sentences, hs=1, min_count=1, window=3, size=100)
model.save('/content/drive/Shareddrives/zhangzhijie5454@gmail.com/file/word2vec/word2vec.model')  # 保存模型


# 加载模型
model = word2vec.Word2Vec.load('/content/drive/Shareddrives/zhangzhijie5454@gmail.com/file/word2vec/word2vec.model')  # 加载模型
for val in model.wv.similar_by_word("足球", topn=10):
    print(val[0], val[1])
    pass


# 进一步测试
model = word2vec.Word2Vec.load('/content/drive/Shareddrives/zhangzhijie5454@gmail.com/file/word2vec/word2vec.model')  # 加载模型
y1 = model.similarity(u"足球", u"足球")
y2 = model.wv.similarity(u"足球", u"足球")
y3 = model.wv.similarity(u'心签名', u'暂停')
y4 = model.similarity(u'心签名', u'暂停')
result = '{}\n{}\n{}\n{}'.format(y1, y2, y3, y4)
print(result)



###################################################################################################################################
import json
from gensim.models import word2vec
from tqdm import tqdm

input_path = '/content/drive/Shareddrives/zhangzhijie5454@gmail.com/file/ch_wordsim_297.json'
output_path = '/content/drive/Shareddrives/zhangzhijie5454@gmail.com/file/word_embeddings.json'

model = word2vec.Word2Vec.load('/content/drive/Shareddrives/zhangzhijie5454@gmail.com/file/word2vec/word2vec.model')

# get the word embedding matrix
words_file = open(input_path, 'r', encoding='utf-8')
result_file = open(output_path, 'a', encoding='utf-8')
word_dict = {}

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

for line in tqdm(words_file.readlines()):
    json_line = json.loads(line)
    word_dict[str(json_line['word1'])] = model[str(json_line['word1'])]
    word_dict[str(json_line['word2'])] = model[str(json_line['word2'])]

result_file.write(json.dumps(word_dict, ensure_ascii=False, cls=NpEncoder))

###################################################################################################################################


###################################################################################################################################
import json
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)
word_embeddings = json.load(open('/content/drive/Shareddrives/zhangzhijie5454@gmail.com/file/word_embeddings.json'))
print(word_embeddings['足球'])
###################################################################################################################################




###################################################################################################################################
import json
import numpy as np
input_path = '/content/drive/Shareddrives/zhangzhijie5454@gmail.com/file/wordsim297.json'
output_path = '/content/drive/Shareddrives/zhangzhijie5454@gmail.com/file/similarity.json'
model_path = '/content/drive/Shareddrives/zhangzhijie5454@gmail.com/file/word2vec/word2vec.model'

# load the word2vec model
model = word2vec.Word2Vec.load(model_path)

# predict the similarity between word1 and word2, save the predict result
words_file = open(input_path, 'r', encoding='utf-8')
result_file = open(output_path, 'a', encoding='utf-8')
for line in words_file.readlines():
    json_line = json.loads(line)
    json_line['similarity'] = str(model.similarity(json_line['word1'], json_line['word2']))
    result_file.write(json.dumps(json_line, ensure_ascii=False) + '\n')
###################################################################################################################################




###################################################################################################################################
from gensim.models import word2vec
model = word2vec.Word2Vec.load('/content/drive/Shareddrives/zhangzhijie5454@gmail.com/file/word2vec/word2vec.model')
l = list(model.wv.vocab.keys())[:10]
print(l)
embedding_matrix = model.wv.vectors
print(embedding_matrix)
###################################################################################################################################
