import os
from tqdm import tqdm
import pandas as pd

#경로 지정
os.chdir("./KoreanChatbot/preprocessing/data_augment/subtitles_html/conv")

#전처리할 파일 이름 입력 후 파일 읽기
data_input = input('write file name:')
data_lines = open(data_input, 'r').readlines()

#0과 1을 입력해서 연결 여부를 입력하세요
df_ = pd.DataFrame(columns = ['sent_1','sent_2','connection'])
for L in tqdm(range(len(data_lines)-1)):
    sent1, sent2 = data_lines[L], data_lines[L+1]
    query = input(sent1+ ' ' + sent2)
    df_ = df_.append({'sent_1':sent1, 'sent_2':sent2, 'connection':query},ignore_index = True)
print(df_)