# -*- coding: utf-8 -*-
import tensorflow as tf
# Define FLAGS
DEFINES = tf.compat.v1.flags.FLAGS

tf.compat.v1.flags.DEFINE_integer('batch_size_transformer', 64, 'batch size')  # 배치 크기
tf.compat.v1.flags.DEFINE_integer('train_steps_transformer', 20000, 'train steps')  # 학습 에포크
tf.compat.v1.flags.DEFINE_float('dropout_width_transformer', 0.5, 'dropout width')  # 드롭아웃 크기
tf.compat.v1.flags.DEFINE_integer('embedding_size_transformer', 128, 'embedding size')  # 임베딩 크기
tf.compat.v1.flags.DEFINE_float('learning_rate_transformer', 1e-3, 'learning rate')  # 학습률
tf.compat.v1.flags.DEFINE_integer('shuffle_seek_transformer', 1000, 'shuffle random seek')  # 셔플 시드값
tf.compat.v1.flags.DEFINE_integer('max_sequence_length_transformer', 25, 'max sequence length')  # 시퀀스 길이
tf.compat.v1.flags.DEFINE_integer('model_hidden_size_transformer', 128, 'model weights size')  # 모델 가중치 크기
tf.compat.v1.flags.DEFINE_integer('ffn_hidden_size_transformer', 512, 'ffn weights size')  # ffn 가중치 크기
tf.compat.v1.flags.DEFINE_integer('attention_head_size_transformer', 4, 'attn head size')  # 멀티 헤드 크기
tf.compat.v1.flags.DEFINE_integer('layer_size_transformer', 2, 'layer size')  # 논문은 6개 레이어이나 2개 사용 학습 속도 및 성능 튜닝
tf.compat.v1.flags.DEFINE_string('data_path_transformer', '../data_in/ChatBotData.csv', 'data path')  # 데이터 위치
tf.compat.v1.flags.DEFINE_string('vocabulary_path_transformer', '../transformer/data_out/vocabularyData.voc', 'vocabulary path')  # 사전 위치
tf.compat.v1.flags.DEFINE_string('check_point_path_transformer', '../transformer/data_out/check_point', 'check point path')  # 체크 포인트 위치
tf.compat.v1.flags.DEFINE_boolean('tokenize_as_morph_transformer', False, 'set morph tokenize')  # 형태소에 따른 토크나이징 사용 유무
tf.compat.v1.flags.DEFINE_boolean('xavier_initializer_transformer', True, 'set xavier initializer')  # 형태소에 따른 토크나이징 사용 유무


