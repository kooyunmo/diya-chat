# converts kakaotalk files from any platforms (iOS, Android, macOS, Windows) to csv format similar to macOS extracted file.

'''
Features and Limitation

Features:
1. convert ios or android text to csv, removing invite, join, left messages
2. 

Limitaitons:
1. in iOS and Android, if username contains space-colon-space(' : '), then it causes problem.
'''
import pandas as pd
import csv
import re
import datetime
import os
import time

savepath = 'converted/'
if not os.path.isdir(savepath):
    os.mkdir(savepath)

get_filename = lambda filepath: filepath.split('/')[-1]

def write_data(filename, data):
    with open(os.path.join(savepath, filename+'.csv'), 'w', encoding='utf-8') as fw:
        csv_fw = csv.writer(fw, quotechar='"', quoting=csv.QUOTE_ALL)
        csv_fw.writerow(['Date', 'User', 'Message'])
        for d in data:
            csv_fw.writerow(d)

def macOS2csv(filepath):
    # remove '님이 들어왔습니다', '님이 나갔습니다', which are not messages
    filename = get_filename(filepath)
    special_text = ['님이 들어왔습니다.', "님이 나갔습니다.", "님을 초대했습니다."]
    with open(filepath, encoding="utf-8") as fr:
        csv_fr = csv.reader(fr)
        with open(os.path.join(savepath, filename+'.csv'), 'w', encoding="utf-8") as fw:
            csv_fw = csv.writer(fw, quotechar='"', quoting=csv.QUOTE_ALL)
            for line in csv_fr:
                message = line[2]
                if not any(text in message for text in special_text):
                    # message is not join or left message
                    csv_fw.writerow(line)


def iOS2csv(filepath):
    # 한국어 버전 (영어의 경우 시간 표현 방식이 다름)
    def is_chat_text_start(text):
        return len(re.findall('\d{4}. \d{1,2}. \d{1,2}. \w{2} \d{1,2}:\d{1,2}, ', text)) > 0

    def is_chat_text_cont(text):
        # is continued chat text if it is not
        # 1. start of chat text
        # 2. things like invite, which contains date and time in slightly different way
        # 3. date text
        return not is_chat_text_start(text) and len(re.findall('\d{4}. \d{1,2}. \d{1,2}. \w{2} \d{1,2}:\d{1,2}:', text)) == 0 and not is_date_text(text)

    def is_date_text(text):
        return bool(re.match('\d{4}년 \d{1,2}월 \d{1,2}일 \w{3}$', text))

    def get_username(text):
        return text.split(',')[1].split(' : ')[0].strip()

    def get_time(text):
        datetime_str = re.findall('\d{4}. \d{1,2}. \d{1,2}. \w{2} \d{1,2}:\d{1,2}', text)[0]
        datetime_split = datetime_str.split()
        year = int(datetime_split[0][:-1])
        month = int(datetime_split[1][:-1])
        day = int(datetime_split[2][:-1])
        ampm = datetime_split[3]
        hour = int(datetime_split[-1].split(':')[0])
        if ampm == '오후' and hour < 12:
            hour += 12
        minute = int(datetime_split[-1].split(':')[-1])

        return datetime.datetime(year, month, day, hour, minute)

    def get_message(text):
        return ' : '.join(text.split(' : ')[1:]).strip()

    filename = get_filename(filepath)
    special_text = ['님이 들어왔습니다.', "님이 나갔습니다.", "님을 초대했습니다."]
    data = []
    with open(filepath, encoding="utf-8") as fr:
        for line in fr:
            line = line.strip()
            if any(text in line for text in special_text):
                continue
            elif line == '':
                # blank line
                continue
            if is_chat_text_start(line):
                time = get_time(line)
                time_str = time.strftime('%Y-%m-%d %H:%M:%S')
                user = get_username(line)
                message = get_message(line)
                data.append([time_str, user, message])
            elif is_chat_text_cont(line):
                if len(data) > 0:
                    data[-1][2] += "\n" + line

    write_data(filename, data)


def Android2csv(filepath):
    # remove '님이 들어왔습니다', '님이 나갔습니다', which are not messages
    # 한국어 버전 (영어의 경우 시간 표현 방식이 다름)
    def is_chat_text_start(text):
        return len(re.findall('\d{4}년 \d{1,2}월 \d{1,2}일 \w{2} \d{1,2}:\d{1,2}, ', text)) > 0 and ':' in text

    def is_chat_text_cont(text):
        # is continued chat text if it is not
        # 1. start of chat text
        # 2. things like invite, which contains date and time in slightly different way
        # 3. date text
        return not is_chat_text_start(text) and len(re.findall('\d{4}년 \d{1,2}월 \d{1,2}일 \w{2} \d{1,2}:\d{1,2}, ', text)) == 0 and not is_date_text(text)

    def is_date_text(text):
        return bool(re.match('\d{4}년 \d{1,2}월 \d{1,2}일 \w{2} \d{1,2}:\d{1,2}$', text))

    def get_username(text):
        return text.split(',')[1].split(' : ')[0].strip()

    def get_time(text):
        datetime_str = re.findall('\d{4}년 \d{1,2}월 \d{1,2}일 \w{2} \d{1,2}:\d{1,2}', text)[0]
        datetime_split = datetime_str.split()
        year = int(datetime_split[0][:-1])
        month = int(datetime_split[1][:-1])
        day = int(datetime_split[2][:-1])
        ampm = datetime_split[3]
        hour = int(datetime_split[-1].split(':')[0])
        if ampm == '오후' and hour < 12:
            hour += 12
        minute = int(datetime_split[-1].split(':')[-1])

        return datetime.datetime(year, month, day, hour, minute)

    def get_message(text):
        return ' : '.join(text.split(' : ')[1:]).strip()

    filename = get_filename(filepath)
    special_text = ['님이 들어왔습니다.', "님이 나갔습니다.", "님을 초대했습니다."]
    data = []
    with open(filepath, encoding="utf-8") as fr:
        for line in fr:
            line = line.strip()
            if any(line.endswith(text) for text in special_text):
                continue
            elif line == '':
                # blank line
                continue
            if is_chat_text_start(line):
                time = get_time(line)
                time_str = time.strftime('%Y-%m-%d %H:%M:%S')
                user = get_username(line)
                message = get_message(line)
                data.append([time_str, user, message])
            elif is_chat_text_cont(line):
                if len(data) > 0:
                    data[-1][2] += "\n" + line

    write_data(filename, data)


def windows2csv(filepath):
    # in Windows, messages have format different from invite/join/left, etc
    # ex. [user] [time] message
    # however, date is not specified for each message so this need to be stored.
    def is_message(text):
        return bool(re.match('\[.+\] \[\w{2} \d{1,2}:\d{2}\] .+', text))

    def parse_message(text):
        inside_square_brackets = re.findall(r'\[.+?\]', text)
        user = inside_square_brackets[0].strip('[]')  # remove brackets at each end
        time = inside_square_brackets[1].strip('[]')
        ampm = time.split()[0]
        hour = int(time.split()[1].split(':')[0])
        minute = int(time.split()[1].split(':')[1])
        if ampm == '오후' and hour < 12:
            hour += 12
        message_start = text.index(']', text.index(']') + 1) + 1
        message = text[message_start:].strip()
        return user, hour, minute, message

    def is_date(text):
        return bool(re.match('--------------- \d{4}년 \d{1,2}월 \d{1,2}일 \w{3} ---------------', text))

    def parse_date(text):
        date = re.findall('\d{4}년 \d{1,2}월 \d{1,2}일', text)[0]
        year = int(date.split()[0][:-1])
        month = int(date.split()[1][:-1])
        day = int(date.split()[2][:-1])

        return year, month, day

    filename = get_filename(filepath)
    special_text = ['님이 들어왔습니다.', "님이 나갔습니다.", "님을 초대하였습니다."]
    data = []
    with open(filepath, encoding="utf-8") as fr:
        year = 0
        month = 0
        day = 0
        for line in fr:
            line = line.strip()
            if is_message(line):
                user, hour, minute, message = parse_message(line)
                time_str = datetime.datetime(year, month, day, hour, minute).strftime('%Y-%m-%d %H:%M:%S')
                data.append([time_str, user, message])

            elif is_date(line):
                year, month, day = parse_date(line)

            elif not any(text in line for text in special_text):
                # continued text, maybe...
                if len(data) > 0:
                    data[-1][2] += '\n' + line

    write_data(filename, data)


def detect_platform_1(filepath):
    # automatically detect platform of kakaotalk text file
    if filepath.endswith('.csv'):
        return 'macOS'
    with open(filepath, encoding="utf-8") as fr:
        lines = [fr.readline() for i in range(10)]

    if 'Date Saved' in lines[1] or 'Saved Date' in lines[1]:
        return 'English'

    for line in lines:
        line = line.strip()
        if re.match('--------------- \d{4}년 \d{1,2}월 \d{1,2}일 \w{3} ---------------', line):
            return 'Windows'

    # android or ios
    first_line = lines[0].strip()
    if first_line.endswith('님과 카카오톡 대화'):
        return 'Android'
    else:
        return 'iOS'

def detect_platform_2(filepath):
    # another method to detect platform is by counting blank lines.
    # Windows has 1, Android has 2, iOS has 3
    if filepath.endswith('.csv'):
        return 'macOS'

    with open(filepath, encoding="utf-8") as fr:
        lines = [fr.readline() for i in range(5)]

    if 'Date Saved' in lines[1] or 'Saved Date' in lines[1]:
        return 'English'

    blanks = sum([line.strip() == '' for line in lines])
    print(blanks)
    if blanks == 1:
        return 'Windows'
    elif blanks == 2:
        return 'Android'
    elif blanks == 3:
        return 'iOS'


def kakao2csv(filepath):
    start = time.time()
    platform = detect_platform_2(filepath)
    print(platform)
    if platform == 'iOS':
        iOS2csv(filepath)
    elif platform == 'macOS':
        macOS2csv(filepath)
    elif platform == 'Android':
        Android2csv(filepath)
    elif platform == 'Windows':
        windows2csv(filepath)
    elif platform == 'English':
        print('No English plz...')
    print(time.time() - start, 's')


if __name__ == '__main__':
    import sys
    filepath = sys.argv[1]
    kakao2csv(filepath)

    
