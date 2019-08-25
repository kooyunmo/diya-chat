from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from django.utils.safestring import mark_safe
from django.urls import reverse

from .models import Room

import json

import random
import tensorflow as tf
import os
import sys


###### import for transformer model inferencing ######
from .transformer import data as tranformer_data
from .transformer import model as transformer
from .transformer.configs import DEFINES as DEFINES_transformer

###### import for seq2seq model inferencing #######
from .seq2seq import data as seq2seq_data
from .seq2seq import model as seq2seq
from .seq2seq.configs import DEFINES as DEFINES_seq2seq



def index(request):
    chatroom_list = Room.objects.order_by('rank')[:3]
    context = {
        'chatroom_list': chatroom_list
    }
    return render(request, 'chatroom/index.html', context)


def room(request, lm_name):
    chatroom_list = Room.objects.order_by('rank')[:3]

    context = {
        'lm_name': mark_safe(json.dumps(lm_name)),
        'chatroom_list': chatroom_list
    }
    return render(request, 'chatroom/room.html', context)


def detail(request, lm_name):

    return HttpResponse("You're looking at chatroom using %s." % lm_name)


def message(request, message, lm_name):
    # if lm_name == 'tranformer':
    # if lm_name == 'seq2seq':
    # if lm_name == 'bert':
    #return HttpResponse("answer: %s" % (message))

    if lm_name == 'transformer':
        print(lm_name)

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


        tf.logging.set_verbosity(tf.logging.ERROR)
        arg_length = len(sys.argv)

        if (arg_length < 2):
            raise Exception("Don't call us. We'll call you")

        # 데이터를 통한 사전 구성 한다.
        char2idx, idx2char, vocabulary_length = tranformer_data.load_vocabulary()

        # 테스트용 데이터 만드는 부분이다.
        # 인코딩 부분 만든다.
        input = message

        print(input)
        predic_input_enc, predic_input_enc_length = tranformer_data.enc_processing([
                                                                        input], char2idx)
        # 학습 과정이 아니므로 디코딩 입력은
        # 존재하지 않는다.(구조를 맞추기 위해 넣는다.)
        predic_output_dec, predic_output_dec_length = tranformer_data.dec_output_processing([
                                                                                            ""], char2idx)
        # 학습 과정이 아니므로 디코딩 출력 부분도
        # 존재하지 않는다.(구조를 맞추기 위해 넣는다.)
        predic_target_dec = tranformer_data.dec_target_processing([
                                                                  ""], char2idx)

        # 에스티메이터 구성한다.
        classifier = tf.estimator.Estimator(
            model_fn=transformer.Model,  # 모델 등록한다.
            # 체크포인트 위치 등록한다.
            model_dir=DEFINES_transformer.check_point_path_transformer,
            params={  # 모델 쪽으로 파라메터 전달한다.
                'embedding_size': DEFINES_transformer.embedding_size_transformer,
                # 가중치 크기 설정한다.
                'model_hidden_size': DEFINES_transformer.model_hidden_size_transformer,
                'ffn_hidden_size': DEFINES_transformer.ffn_hidden_size_transformer,
                'attention_head_size': DEFINES_transformer.attention_head_size_transformer,
                # 학습율 설정한다.
                'learning_rate': DEFINES_transformer.learning_rate_transformer,
                # 딕셔너리 크기를 설정한다.
                'vocabulary_length': vocabulary_length,
                # 임베딩 크기를 설정한다.
                'embedding_size': DEFINES_transformer.embedding_size_transformer,
                'layer_size': DEFINES_transformer.layer_size_transformer,
                'max_sequence_length': DEFINES_transformer.max_sequence_length_transformer,
                'xavier_initializer': DEFINES_transformer.xavier_initializer_transformer
            })

        answer = ""

        for i in range(25):
            if i > 0:
                predic_output_dec, predic_output_decLength = tranformer_data.dec_output_processing([
                                                                                                   answer], char2idx)
                predic_target_dec = tranformer_data.dec_target_processing(
                    [answer], char2idx)
            # 예측을 하는 부분이다.
            predictions = classifier.predict(
                input_fn=lambda: tranformer_data.eval_input_fn(predic_input_enc, predic_output_dec, predic_target_dec, 1))

            answer, finished = tranformer_data.pred_next_string(
                predictions, idx2char)

            if finished:
                break

        print("answer: ", answer)



    elif lm_name == 'seq2seq':
        print(lm_name)

        tf.logging.set_verbosity(tf.logging.INFO)
        arg_length = len(sys.argv)

        if(arg_length < 2):
            raise Exception("Don't call us. We'll call you")

        char2idx,  idx2char, vocabulary_length = seq2seq_data.load_vocabulary()
        input = message

        print(input)
        # 테스트셋 인코딩 / 디코딩 입력 / 디코딩 출력 만드는 부분이다.
        predic_input_enc, predic_input_enc_length = seq2seq_data.enc_processing([
                                                                        input], char2idx)
        predic_output_dec, predic_output_dec_length = seq2seq_data.dec_input_processing([
                                                                                ""], char2idx)
        predic_target_dec = seq2seq_data.dec_target_processing([""], char2idx)

        # 에스티메이터 구성
        classifier = tf.estimator.Estimator(
            model_fn=seq2seq.model,
            model_dir=DEFINES_seq2seq.check_point_path_seq2seq,
            params={
                'hidden_size': DEFINES_seq2seq.hidden_size_seq2seq,
                'layer_size': DEFINES_seq2seq.layer_size_seq2seq,
                'learning_rate': DEFINES_seq2seq.learning_rate_seq2seq,
                'vocabulary_length': vocabulary_length,
                'embedding_size': DEFINES_seq2seq.embedding_size_seq2seq,
                'embedding': DEFINES_seq2seq.embedding_seq2seq,
                'multilayer': DEFINES_seq2seq.multilayer_seq2seq,
            })

        predictions = classifier.predict(
            input_fn=lambda: seq2seq_data.eval_input_fn(predic_input_enc, predic_output_dec, predic_target_dec, DEFINES_seq2seq.batch_size_seq2seq))

        answer = seq2seq_data.pred2string(predictions, idx2char)


    return HttpResponse("%s" % answer)


# this is only for test
'''
def helloworld(request, num, lm_name):
    n1 = int(num)
    n2 = 1
    x = tf.placeholder(tf.int32)
    y = tf.placeholder(tf.int32)

    add = tf.add(x, y)
    with tf.Session() as sess:
        z = sess.run(add, feed_dict={x:n1, y:n2})

    return HttpResponse("sum of %s and 1 is %s" %(num, z))
'''
