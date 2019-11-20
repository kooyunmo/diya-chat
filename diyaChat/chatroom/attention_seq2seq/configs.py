#*- coding: utf-8 -*-
import tensorflow as tf

# Define FLAGS
DEFINES = tf.compat.v1.flags.FLAGS

tf.compat.v1.flags.DEFINE_integer('batch_size_attention_seq2seq', 64, 'batch size') # 배치 크기
tf.compat.v1.flags.DEFINE_integer(
    'train_steps_attention_seq2seq', 30000, 'train steps')  # 학습 에포크
tf.compat.v1.flags.DEFINE_float(
    'dropout_width_attention_seq2seq', 0.8, 'dropout width')  # 드롭아웃 크기
tf.compat.v1.flags.DEFINE_integer(
    'layer_size_attention_seq2seq', 1, 'layer size')  # 멀티 레이어 크기 (multi rnn)
tf.compat.v1.flags.DEFINE_integer(
    'hidden_size_attention_seq2seq', 128, 'weights size')  # 가중치 크기
tf.compat.v1.flags.DEFINE_float(
    'learning_rate_attention_seq2seq', 1e-3, 'learning rate')  # 학습률
tf.compat.v1.flags.DEFINE_float('teacher_forcing_rate_attention_seq2seq',
                          0.7, 'teacher forcing rate')  # 학습시 디코더 인풋 정답 지원율
tf.compat.v1.flags.DEFINE_string('data_path_attention_seq2seq',
                           '../data_in/ChatBotData.csv', 'data path')  # 데이터 위치
tf.compat.v1.flags.DEFINE_string(
    'vocabulary_path_attention_seq2seq', '../attention_seq2seq/data_out/vocabularyData.voc', 'vocabulary path')  # 사전 위치
tf.compat.v1.flags.DEFINE_string(
    'check_point_path_attention_seq2seq', '../attention_seq2seq/data_out/check_point', 'check point path')  # 체크 포인트 위치
tf.compat.v1.flags.DEFINE_string(
    'save_model_path_attention_seq2seq', '../attention_seq2seq/data_out/model', 'save model')  # 모델 저장 경로
tf.compat.v1.flags.DEFINE_integer(
    'shuffle_seek_attention_seq2seq', 1000, 'shuffle random seek')  # 셔플 시드값
tf.compat.v1.flags.DEFINE_integer(
    'max_sequence_length_attention_seq2seq', 25, 'max sequence length')  # 시퀀스 길이
tf.compat.v1.flags.DEFINE_integer(
    'embedding_size_attention_seq2seq', 128, 'embedding size')  # 임베딩 크기
tf.compat.v1.flags.DEFINE_boolean(
    'embedding_attention_seq2seq', True, 'Use Embedding flag')  # 임베딩 유무 설정
tf.compat.v1.flags.DEFINE_boolean(
    'multilayer_attention_seq2seq', True, 'Use Multi RNN Cell')  # 멀티 RNN 유무
tf.compat.v1.flags.DEFINE_boolean(
    'attention_attention_seq2seq', True, 'Use Attention')  # 어텐션 사용 유무
tf.compat.v1.flags.DEFINE_boolean('teacher_forcing_attention_seq2seq',
                            True, 'Use Teacher Forcing')  # 학습시 디코더 인풋 정답 지원 유무
tf.compat.v1.flags.DEFINE_boolean('tokenize_as_morph_attention_seq2seq',
                            False, 'set morph tokenize')  # 형태소에 따른 토크나이징 사용 유무
tf.compat.v1.flags.DEFINE_boolean(
    'serving_attention_seq2seq', False, 'Use Serving')  # 서빙 기능 지원 여부
tf.compat.v1.flags.DEFINE_boolean('loss_mask_attention_seq2seq', False, 'Use loss mask')   # PAD에 대한 마스크를 통한 loss를 제한


