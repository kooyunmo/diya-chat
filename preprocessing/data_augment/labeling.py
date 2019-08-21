import os
from tqdm import tqdm
import pickle

#경로 지정
data_path = "C:/Users/임상규/Documents/NLP/KoreanChatbot/preprocessing/data_augment/subtitles_html"
os.chdir(data_path)

#전처리할 파일 이름 입력 후 파일 읽기
# loader
while True:   
    try:
        folder = input("Which folder? ('conv' for new file, 'labeling' for existing file):")    
        data_input = input('write file name:') + '.txt'
        data_lines = open(data_path + '/'+folder+ '/' + data_input, 'r').readlines()
        break
        
    except:
        print('Try again')
        continue

#새로운 레이블링 파일 만들기
if not os.path.isfile(data_path+'/labeling/L_'+data_input):
    for L in tqdm(range(len(data_lines)-1)):
        sent1, sent2 = data_lines[L].rstrip(), data_lines[L+1].rstrip()
        print('두 문장의 관계는? \n 0: 이어지지 않음, 1:이어짐, 2: 한 문장, 3: 쓸모없음')
        query = input(sent1+ '\n' + sent2+'\n')
        if query == 'q':
            break
        while query not in {'0','1','2','3'}:
            query = input('0,1,2,3 중 입력해주세요!')
        txt = open(data_path + '/labeling/L_' + data_input,'a')
        txt.write(sent1+'[SEP]'+sent2+'[SEP]'+query+'\n')
        txt.close()
        print('Saved')
        
#기존 레이블링 파일에서 마저 작업하고 싶을때
else:
    txt = open(data_path + '/labeling/L_' + data_input,'r').readlines()
    print("기존 파일 수정 중...")
    start = txt[-1].split('[SEP]')[1]
    for L in tqdm(range(data_lines.index(start+'\n'),len(data_lines)-1)):
        sent1, sent2 = data_lines[L].rstrip(), data_lines[L+1].rstrip()
        print('두 문장의 관계는? \n 0: 이어지지 않음, 1:이어짐, 2: 한 문장, 3: 쓸모없음')
        query = input(sent1+ '\n' + sent2+'\n')
        if query == 'q':
            break
        while query not in {'0','1','2','3'}:
            query = input('0,1,2,3 중 입력해주세요!')
        txt = open(data_path + '/labeling/L_' + data_input,'a')
        txt.write(sent1+'[SEP]'+sent2+'[SEP]'+query+'\n')
        txt.close()
        print('Saved')