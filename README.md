# KoreanChatbot
Korean chatbot with various language models

---

## 1. Overview

- This project is written by both tensorflow and pytorch frameworks.
- The models are trained by Korean QA-pair chatbot data.
    + The data is provided by this github link: https://github.com/songys/Chatbot_data
- We tried to train various language model using this Korean chatbot data.

### LMs used in this project

1. BERT(by fine-tuning pretrained model: KorBERT)
2. Transformer (based on Attention is all you need)

### Data

- We create new Q-A data set with the movie subtitles
- The movie subtitle data came from https://www.gomlab.com/subtitle/


---

* 파이썬 가상환경 설정(https://tutorial.djangogirls.org/ko/django_installation/)


* 패키지 관리
  - 패키지 설치
    ```
    pip install -r requirements.txt
    sudo pip install [package_name] --upgrade   # 패키지 업데이트
    ```
  - 현재 패키지 설정 requirements.txt에 기록
    ```
    pip freeze > requirements.txt
    ```
* docker redis server 실행 (채팅 기능)
  ```
  # 설치
  brew install Docker 후에 홈페이지에서 도커 데스크탑 파일을 다운받아서 설치
  docker version으로 설치 확인
  docker pull redis

  # 실행
  sudo docker run -p 6379:6379 -d redis
  ```
* django server 실행
  ```
  python manage.py runserver
  ```

* DB Browser for sqlite 사용하여 DB 관리
