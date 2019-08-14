from bs4 import BeautifulSoup

import numpy as np
import pandas as pd
import os
import re

DATA_IN_PATH = './subtitles_html/'
DATA_OUT_PATH = './subtitles_html/conv/'

print("file name".ljust(30) + "size")
for file in os.listdir(DATA_IN_PATH):
    if 'html' in file:
        print(file.ljust(
            30) + str(round(os.path.getsize(DATA_IN_PATH + file) / 1000000, 2)) + 'MB')

    try:
        with open(os.path.join(DATA_IN_PATH, file), 'r') as f:
            html_string = f.read()

        soup = BeautifulSoup(html_string)

        html_string = html_string.replace('<br>', ' ')

        conv_string = re.sub("\<.*?\>|&nbsp;|sub2smi", "", html_string)
        conv_string = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ,.?!\s]", "", conv_string)

        conv_string = conv_string.split(sep='\n')

        str_list = list(filter(None, conv_string))

        with open(os.path.join(DATA_OUT_PATH, file), 'w+') as f:
            for item in str_list:
                f.write("%s\n" % item)

    except UnicodeDecodeError:
        print("This file is not 'utf-8' format. Skip this file.")
        pass

    except IsADirectoryError:
        pass

    except PermissionError:
       pass
