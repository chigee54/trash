from gensim.models import word2vec
from flask import Flask
from flask_restful import Api, Resource
import json

app = Flask(__name__)
app.config.update(RESTFUL_JSON=dict(ensure_ascii=False))
api = Api(app)
output_vec_path = './output/word_vectors.json'
model_path = './model/word2vec.model'

print("Loading the model...")
model = word2vec.Word2Vec.load(model_path)


def word2vec_model(input_json, model, input_string=True):
    print("The model has been loaded...doing predictions now...")
    word_dict = {}
    if input_string:
        word_dict[str(input_json)] = model.wv[str(input_json)].tolist()
        vec_result_file = open(output_vec_path, 'w', encoding='utf-8')
        vec_result_file.write(json.dumps(word_dict, ensure_ascii=False))
    else:
        word_dict[str(input_json['word1'])] = model.wv[str(input_json['word1'])].tolist()
        word_dict[str(input_json['word2'])] = model.wv[str(input_json['word2'])].tolist()
        vec_result_file = open(output_vec_path, 'w', encoding='utf-8')
        vec_result_file.write(json.dumps(word_dict, ensure_ascii=False))

    return word_dict


class INPUT(Resource):
    def get(self, word_id='all'):
        input_file = open('./input/ch_wordsim_297.json', 'r', encoding='utf-8')
        json_file = input_file.readlines()
        if word_id == 'all':
            return json_file
        else:
            word_pair = json_file[int(word_id) - 1]
            show = 'Total: {}, ID: {}, Word Pair: {}'.format(len(json_file), word_id, word_pair)
            return show


class OUTPUT(Resource):
    def get(self, word_id):
        input_file = open('./input/ch_wordsim_297.json', 'r', encoding='utf-8')
        json_file = input_file.readlines()
        if word_id == 'all':
            output_list = []
            for i in json_file:
                output = word2vec_model(json.loads(i), model, False)
                output_list.append(output)
            return output_list
        else:
            word_pair = json.loads(json_file[int(word_id) - 1])
            output = word2vec_model(word_pair, model, False)
            show = 'Total: {}, ID: {}, Output: {}'.format(len(json_file), word_id, output)
            return show


class WORD_OUTPUT(Resource):
    def get(self, word):
        output = word2vec_model(word, model)
        show = 'Output: {}'.format(output)
        return show


api.add_resource(INPUT, '/GF3.1/json-input/<word_id>')
api.add_resource(WORD_OUTPUT, '/GF3.1/word-output/<word>')
api.add_resource(OUTPUT, '/GF3.1/json-output/<word_id>')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000')
