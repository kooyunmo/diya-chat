#-*- coding: utf-8 -*-
import tensorflow as tf

# Define FLAGS
DEFINES = tf.compat.v1.flags.FLAGS

tf.compat.v1.flags.DEFINE_string('f', '', 'kernel') # 주피터에서 커널에 전달하기 위한 프레그 방법
tf.compat.v1.flags.DEFINE_integer('batch_size_seq2seq', 64, 'batch size') # 배치 크기
tf.compat.v1.flags.DEFINE_integer('train_steps_seq2seq',
                            20000, 'train steps')  # 학습 에포크
tf.compat.v1.flags.DEFINE_float('dropout_width_seq2seq',
                          0.5, 'dropout width')  # 드롭아웃 크기
tf.compat.v1.flags.DEFINE_integer('layer_size_seq2seq', 3,
                            'layer size')  # 멀티 레이어 크기 (multi rnn)
tf.compat.v1.flags.DEFINE_integer('hidden_size_seq2seq', 128, 'weights size') # 가중치 크기
tf.compat.v1.flags.DEFINE_float('learning_rate_seq2seq', 1e-3, 'learning rate') # 학습률
tf.compat.v1.flags.DEFINE_string(
    'data_path_seq2seq', '../data_in/ChatBotData.csv', 'data path')  # 데이터 위치
tf.compat.v1.flags.DEFINE_string('vocabulary_path_seq2seq',
                           '../seq2seq/data_out/vocabularyData.voc', 'vocabulary path')  # 사전 위치
tf.compat.v1.flags.DEFINE_string('check_point_path_seq2seq',
                           '../seq2seq/data_out/check_point', 'check point path')  # 체크 포인트 위치
tf.compat.v1.flags.DEFINE_integer('shuffle_seek_seq2seq',
                            1000, 'shuffle random seek')  # 셔플 시드값
tf.compat.v1.flags.DEFINE_integer(
    'max_sequence_length_seq2seq', 25, 'max sequence length')  # 시퀀스 길이
tf.compat.v1.flags.DEFINE_integer(
    'embedding_size_seq2seq', 128, 'embedding size')  # 임베딩 크기
tf.compat.v1.flags.DEFINE_boolean(
    'tokenize_as_morph_seq2seq', True, 'set morph tokenize')  # 형태소에 따른 토크나이징 사용 유무
tf.compat.v1.flags.DEFINE_boolean(
    'embedding_seq2seq', True, 'Use Embedding flag')  # 임베딩 유무 설정
tf.compat.v1.flags.DEFINE_boolean(
    'multilayer_seq2seq', True, 'Use Multi RNN Cell')  # 멀티 RNN 유무
