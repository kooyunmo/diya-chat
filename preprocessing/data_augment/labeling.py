import os
from tqdm import tqdm
import pandas as pd

#경로 지정
os.chdir("./KoreanChatbot/preprocessing/data_augment/subtitles_html/conv")

#전처리할 파일 이름 입력 후 파일 읽기
data_input = input('write file name:')
data_lines = open(data_input, 'r').readlines()
#몇번째 문장부터 전처리를 시작할 것인가(쓸데없는 문장들 무시)
start = int(input('Enter index of sentence to start from:'))

df_ = pd.DataFrame(columns = ['sent_1','sent_2','connection'])
for L in tqdm(range(start,len(data_lines)-1)):
    sent1, sent2 = data_lines[L], data_lines[L+1]
    print('두 문장의 관계는? \n 0: 이어지지 않음, 1:이어짐, 2: 한 문장, 3: 쓸모없음 \n')
    query = int(input(sent1+ ' ' + sent2))
    while query != (0 or 1 or 2 or 3):
        print('0,1,2,3 중 입력해주세요!')
        query = int(input())
    df_ = df_.append({'sent_1':sent1, 'sent_2':sent2, 'connection':query},ignore_index = True)

df_.to_pickle("./sent_connection.pkl")