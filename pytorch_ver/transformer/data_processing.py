from konlpy.tag import Okt
import pandas as pd
import torch
import torch.nn as nn
import enum
import os
import re
from sklearn.model_selection import train_test_split
import numpy as np

from tqdm import tqdm

################## global variables ##################

FILTERS = "([~.,!?\"':;)(])"
PAD = "<PAD>"
STD = "<SOS>"
END = "<END>"
UNK = "<UNK>"

PAD_INDEX = 0
STD_INDEX = 1
END_INDEX = 2
UNK_INDEX = 3

MARKER = [PAD, STD, END, UNK]
CHANGE_FILTER = re.compile(FILTERS)

path = '../../data_in/ChatBotData.csv'
vocab_path = '../../data_out/vocab.voc'
data_path = '../../data_in/ChatBotData.csv'
max_sequence_len = 25

############### end of global variables ##############


def load_data():
    '''
    1. Read Korean chatbot data(Q-A pairs) from the path, and put them into pd.DataFrame.
    2. split data into training set and validation set, and return them.

    @return
        - train_input: training question
        - train_label: training answer
        - test_input: test question
        - test_label: test answer
    '''
    data_df = pd.read_csv(path, header=0)
    question, answer = list(data_df['Q']), list(data_df['A'])
    train_input, test_input, train_label, test_label = train_test_split(
        question, answer, test_size=0.33, random_state=42)

    return train_input, train_label, test_input, test_label


# 사용 안 할 예정
def prepro_like_morphlized(data):
    '''
    @params:
        - data:
    '''
    # 형태소 분석 모듈 객체를 생성
    morph_analyzer = Okt()

    # 형태소 토크나이즈 결과 문장을 받을 리스트를 생성합니다.
    result_data = list()
    # 데이터에 있는 매 문장에 대해 토크나이즈를 할 수 있도록 반복문을 선언합니다.
    for seq in tqdm(data):
        # Okt.morphs 함수를 통해 토크나이즈 된 리스트 객체를 받고 다시 공백문자를 기준으로 문자열로 재구성 해줍니다.
        morphlized_seq = " ".join(morph_analyzer.morphs(seq))
        result_data.append(morphlized_seq)

    return result_data


# helper for load_vacabulary()
def data_tokenizer(data):
    # 토크나이징 해서 담을 배열 생성
    words = []
    for sentence in data:
        # FILTERS = "([~.,!?\"':;)(])"
        # 위 필터와 같은 값들을 정규화 표현식을 통해서 모두 "" 으로 변환 해주는 부분이다.
        sentence = re.sub(CHANGE_FILTER, "", sentence)
        for word in sentence.split():
            words.append(word)

    # 토그나이징과 정규표현식을 통해 만들어진 값들을 넘겨 준다.
    return [word for word in words if word]

# helper for load_vacabulary()


def make_vocabulary(vocabulary_list):
    # 리스트를 키가 단어이고 값이 인덱스인 딕셔너리를 만든다.
    char2idx = {char: idx for idx, char in enumerate(vocabulary_list)}
    # 리스트를 키가 인덱스이고 값이 단어인 딕셔너리를 만든다.
    idx2char = {idx: char for idx, char in enumerate(vocabulary_list)}
    # 두개의 딕셔너리를 넘겨 준다.
    return char2idx, idx2char


def load_vocabulary():
    # 사전을 담을 배열 준비한다.
    vocabulary_list = []

    # 사전을 구성한 후 파일로 저장 진행한다.
    # 그 파일의 존재 유무를 확인한다.
    if (not (os.path.exists(vocab_path))):
        # 이미 생성된 사전 파일이 존재하지 않으므로 데이터를 가지고 만들어야 한다.
        # 데이터가 존재 하면 사전을 만들기 위해서 데이터 파일의 존재 유무를 확인한다.
        if (os.path.exists(data_path)):
            # 데이터가 존재하면 pandas를 통해서 데이터를 불러오자
            data_df = pd.read_csv(data_path, encoding='utf-8')

            # 판다스의 데이터 프레임을 통해서 질문과 답에 대한 열을 가져 온다.
            question, answer = list(data_df['Q']), list(data_df['A'])

            data = []
            # 질문과 답변을 extend을 통해서 구조가 없는 배열로 만든다.
            data.extend(question)
            data.extend(answer)

            # 토큰나이져 처리 하는 부분이다.
            words = data_tokenizer(data)

            # set으로 중복이 제거된 집합을 생성한 후 리스트로 만들어 준다.
            words = list(set(words))

            # PAD = "<PADDING>"
            # STD = "<START>"
            # END = "<END>"
            # UNK = "<UNKNWON>"
            words[:0] = MARKER

        # 사전 리스트를 사전 파일로 만들어 넣는다.
        with open(vocab_path, 'w', encoding='utf-8') as vocabulary_file:
            for word in words:
                vocabulary_file.write(word + '\n')

    # 사전 파일이 존재하면 여기에서 그 파일을 불러서 배열에 넣어 준다.
    with open(vocab_path, 'r', encoding='utf-8') as vocabulary_file:
        for line in vocabulary_file:
            vocabulary_list.append(line.strip())

    # 배열에 내용을 키와 값이 있는 딕셔너리 구조로 만든다.
    char2idx, idx2char = make_vocabulary(vocabulary_list)

    # 두가지 형태의 키와 값이 있는 형태를 리턴한다.
    # (예) 단어: 인덱스 , 인덱스: 단어)
    return char2idx, idx2char, len(char2idx)


def enc_processing(input_data, dictionary):
    '''
    @params
        - input_data: 인덱싱할 데이터 (train_input 또는 test_input)
        - dictionary: key(word)-value(index) pair

    @return
        - index sequences로 변환된 input_data의 words sequences
            + ex) [13042, 15055, 11881, 12337, 0, 0, ...]
    '''
    # 인덱스 값들을 가지고 있는 배열이다.(누적된다.)
    sequences_input_index = []
    # 하나의 인코딩 되는 문장의 길이를 가지고 있다.(누적된다.)
    sequences_length = []

    # 한줄씩 불어온다.
    for sequence in input_data:
        # FILTERS = "([~.,!?\"':;)(])"
        # 정규화를 사용하여 필터에 들어 있는 값들을 "" 으로 치환 한다.
        sequence = re.sub(CHANGE_FILTER, "", sequence)

        # 하나의 문장을 인코딩 할때 가지고 있기 위한 배열이다.
        sequence_index = []

        # 문장을 스페이스 단위로 자르고 있다.
        for word in sequence.split():
            # 잘려진 단어들이 딕셔너리에 존재 하는지 보고 그 값을 가져와 sequence_index에 추가한다.
            if dictionary.get(word) is not None:
                sequence_index.extend([dictionary[word]])

            # 잘려진 단어가 딕셔너리에 존재 하지 않는 경우 이므로 UNK(2)를 넣어 준다.
            else:
                sequence_index.extend([dictionary[UNK]])

        # 문장 제한 길이보다 길어질 경우 뒤에 토큰을 자르고 있다.
        if len(sequence_index) > max_sequence_len:
            sequence_index = sequence_index[:max_sequence_len]

        # 하나의 문장에 길이를 넣어주고 있다.
        sequences_length.append(len(sequence_index))

        # max_sequence_length보다 문장 길이가 작다면 빈 부분에 PAD(0)를 넣어준다.
        sequence_index += (max_sequence_len -
                           len(sequence_index)) * [dictionary[PAD]]

        # 인덱스화 되어 있는 값을 sequences_input_index에 넣어 준다.
        sequences_input_index.append(sequence_index)

    # 인덱스화된 일반 배열을 넘파이 배열로 변경한다.
    # 이유는 pytorch Variable에 넣어 주기 위한 사전 작업이다.
    # 넘파이 배열에 인덱스화된 배열과 그 길이를 넘겨준다.
    return np.asarray(sequences_input_index), sequences_length


def dec_output_processing(input_data, dictionary):
    '''
    @params
        - input_data: 인덱싱할 데이터 (train_input 또는 test_input)
        - dictionary: key(word)-value(index) pair

    @return
        - index sequences로 변환된 words sequences
    '''

    # 인덱스 값들을 가지고 있는 배열이다.(누적된다)
    sequences_output_index = []

    # 하나의 디코딩 입력 되는 문장의 길이를 가지고 있다.(누적된다)
    sequences_length = []

    # 한줄씩 불어온다.
    for sequence in input_data:
        # FILTERS = "([~.,!?\"':;)(])"
        # 정규식을 사용하여 필터에 들어 있는 값들을 "" 으로 치환 한다.
        sequence = re.sub(CHANGE_FILTER, "", sequence)

        # "하나의 문장"을 디코딩 할때 가지고 있기 위한 배열이다.
        sequence_index = []

        # 디코딩 입력의 처음에는 START(STD)가 와야 하므로 그 값을 넣어 주고 시작한다.
        # 문장에서 스페이스 단위별로 단어를 가져와서 딕셔너리의 값인 인덱스를 넣어 준다.
        sequence_index = [dictionary[STD]] + [dictionary[word]
                                              for word in sequence.split()]

        # 문장 제한 길이보다 길어질 경우 뒤에 토큰을 자르고 있다.
        if len(sequence_index) > max_sequence_len:
            sequence_index = sequence_index[:max_sequence_len]

        # 하나의 문장에 길이를 넣어주고 있다.
        sequences_length.append(len(sequence_index))

        # max_sequence_length보다 문장 길이가 작다면 빈 부분에 PAD(0)를 넣어준다.
        sequence_index += (max_sequence_len -
                           len(sequence_index)) * [dictionary[PAD]]

        # 인덱스화 되어 있는 값을 sequences_output_index 넣어 준다.
        sequences_output_index.append(sequence_index)

    # 인덱스화된 일반 배열을 넘파이 배열로 변경한다.
    # 이유는 pytorch Variable에 넣어 주기 위한 사전 작업이다.
    # 넘파이 배열에 인덱스화된 배열과 그 길이를 넘겨준다.
    return np.asarray(sequences_output_index), sequences_length


def dec_target_processing(input_data, dictionary):
    '''
    @params
        - input_data: 인덱싱할 데이터 (train_input 또는 test_input)
        - dictionary: key(word)-value(index) pair

    @return
        - index sequences로 변환된 words sequences
    '''

    # 인덱스 값들을 가지고 있는 배열이다.(누적된다)
    sequences_target_index = []

    # input_data에서 한줄씩 불어온다.
    for sequence in input_data:
        # FILTERS = "([~.,!?\"':;)(])"
        # 정규식을 사용하여 필터에 들어 있는 값들을 "" 으로 치환 한다.
        sequence = re.sub(CHANGE_FILTER, "", sequence)

        # 문장에서 스페이스 단위별로 단어를 가져와서 딕셔너리의 값인 인덱스를 넣어 준다.
        # 디코딩 출력의 마지막에 END를 넣어 준다.
        sequence_index = [dictionary[word] for word in sequence.split()]

        # 문장 제한 길이보다 길어질 경우 뒤에 토큰을 자르고 END 토큰을 넣어 준다
        if len(sequence_index) >= max_sequence_len:
            sequence_index = sequence_index[:max_sequence_len -
                                            1] + [dictionary[END]]
        else:
            sequence_index += [dictionary[END]]

        # max_sequence_length보다 문장 길이가 작다면 빈 부분에 PAD(0)를 넣어준다.
        sequence_index += (max_sequence_len -
                           len(sequence_index)) * [dictionary[PAD]]
        # 인덱스화 되어 있는 값을 sequences_target_index에 넣어 준다.
        sequences_target_index.append(sequence_index)

    # 인덱스화된 일반 배열을 넘파이 배열로 변경한다.
    # 이유는 pytorch Variable에 넣어 주기 위한 사전 작업이다.
    # 넘파이 배열에 인덱스화된 배열과 그 길이를 넘겨준다.
    return np.asarray(sequences_target_index)


def pred_next_string(index_sequences, dictionary):
    '''
    @params
        - index_sequences: 인덱스로 표현된 sentences
            + ex) [13042, 15055, 11881, 12337, 0, 0, ...] 와 같은 배열 여러 개
        - dictionary: key(index)-value(word) pair

    @return
        - answer: 단어들로 구성된 문장을 반환
        - is_finished: terminate condition
    '''

    # 텍스트 문장을 보관할 배열을 선언한다.
    sentence_string = []
    is_finished = False

    # 인덱스 배열 하나를 꺼내서 v에 넘겨준다.
    for v in index_sequences:
        # 딕셔너리에 있는 단어로 변경해서 배열에 담는다.
        sentence_string = [dictionary[index] for index in v]

    answer = ""
    # 패딩값도 담겨 있으므로 패딩은 모두 스페이스 처리 한다.
    for word in sentence_string:
        if word == END:
            is_finished = True
            break

        if word != PAD and word != END:
            answer += word
            answer += " "

    # 결과를 출력한다.
    return answer, is_finished
