from __future__ import print_function
from flask import Flask, jsonify, request


import os
import tensorflow as tf

from cnn_model import TCNNConfig, TextCNN
from data.word_loader import read_category, read_vocab

unicode = str

base_dir = 'data'
vocab_dir = os.path.join(base_dir, 'word_vocab.txt')

save_dir = 'checkpoints/textcnn'
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径


class CnnModel:
    def __init__(self):
        self.config = TCNNConfig()
        self.categories, self.cat_to_id = read_category()
        self.words, self.word_to_id = read_vocab(vocab_dir)
        self.config.vocab_size = len(self.words)
        self.model = TextCNN(self.config)

        self.session = tf.compat.v1.Session()
        self.session.run(tf.compat.v1.global_variables_initializer())
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess=self.session, save_path=save_path)  # 读取保存的模型

    def predict(self, message):
        # 支持不论在python2还是python3下训练的模型都可以在2或者3的环境下运行
        content = unicode(message)
        data = [self.word_to_id[x] for x in content if x in self.word_to_id]

        feed_dict = {
            self.model.input_x: tf.keras.preprocessing.sequence.pad_sequences([data], self.config.seq_length),
            self.model.keep_prob: 1.0
        }

        y_pred_cls = self.session.run(self.model.y_pred_cls, feed_dict=feed_dict)
        return self.categories[y_pred_cls[0]]

app = Flask(__name__)
app.config['TIMEOUT'] = 60
cnn_model = CnnModel()
@app.route('/predict', methods=['POST'])
def predict():
    # 从请求中获取数据
    input_data = request.form.get('input')
    # 使用模型进行预测
    result = cnn_model.predict([input_data])
    # 返回预测结果
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
# if __name__ == '__main__':
#     cnn_model = CnnModel()
#     input = '我老婆怀孕了但在不知情的情况下，服用了迪康肤痒颗粒这个药，我想问一下影响大不大，到院就诊我该挂哪个科？'
#     test_demo = [input]
#     answer = cnn_model.predict(0)
#     print(answer)
