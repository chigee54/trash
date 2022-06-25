#! -*- coding: utf-8 -*-
# Date: 2021/9/20
# IO for word2vec model
# Usage: python word2vec_io.py --input input/ch_wordsim_297.json
#                              --model model/word2vec.model
#                              --output output/similarity.json

import json
import argparse
from gensim.models import word2vec

parser = argparse.ArgumentParser(description="Add three paths.")
parser.add_argument('--input', type=str, help='input path of the json file.')
parser.add_argument('--output', type=str, help='output path of the similarity result.')
parser.add_argument('--model', type=str, help='the path of word2vec model')
args = parser.parse_args()
input_path = args.input
output_path = args.output
model_path = args.model

# load the word2vec model
model = word2vec.Word2Vec.load(model_path)

# predict the similarity between word1 and word2, save the predict result
words_file = open(input_path, 'r', encoding='utf-8')
result_file = open(output_path, 'a', encoding='utf-8')
for line in words_file.readlines():
    json_line = json.loads(line)
    json_line['similarity'] = str(model.similarity(json_line['word1'], json_line['word2']))
    result_file.write(json.dumps(json_line, ensure_ascii=False) + '\n')

# get the vocabulary
vocab = list(model.wv.vocab.keys())

# get the embedding matrix
embedding_matrix = model.wv.vectors
