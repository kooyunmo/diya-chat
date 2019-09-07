# KoreanChatbot
**Korean chatbot with various language models. This chatbot runs on the local Django website, which will be incorporated to the official [DIYA(Do It Yourself AI) webpage](diyaml.com)**



## 1. Overview

- This project is written by both tensorflow and pytorch frameworks.
- The models are trained by Korean QA-pair chatbot data.
    + The data is provided by this github link: https://github.com/songys/Chatbot_data
- We tried to train various language model using this Korean chatbot data.

#### LMs used in this project

1. BERT(by fine-tuning pretrained model: KorBERT)
2. Transformer (based on Attention is all you need)
3. Sequence to sequence with attention mechanism

#### Data

- We create new Q-A data set with the movie subtitles
- The movie subtitle data came from https://www.gomlab.com/subtitle/


## 2. Install and Run

* How to set virtualenv
  ```
  # Install
  $ sudo apt-get install python3-virtualenv   # for linux
  $ pip install virtualenv                    # for mac / windows

  # Run
  $ virtualenv --python=python3 diyachat-env
  $ source diyachat-env/bin/activate
  ```
  - For more detail: https://tutorial.djangogirls.org/ko/django_installation/


* How to install and set packages
  - Install requirements
    ```
    $ pip install -r requirements.txt
    $ sudo pip install [package_name] --upgrade   # update packages
    ```

  - Save current packages into requirements.txt
    ```
    $ pip freeze > requirements.txt
    ```

* Run docker redis server (for chatting)
  - Installation
    1. Install docker
    ```
    $ brew install Docker
    ```
    2. docker pull redis

  - Run
    ```
      sudo docker run -p 6379:6379 -d redis
    ```
* Run django server
  ```
    $ python manage.py runserver
  ```
  - After entering the command above, go to your web brower and go to the link(http://localhost:8000/)

* Setup sqlite DB
  ```
    $ python manage.py makemigrations
    $ python manage.py migrate
  ```
  - If you want to reset the whole DB, you need to delete `db.sqlite3` file and follow the process above again
  - For more detail check [Djang tutorial: PART2](https://docs.djangoproject.com/ko/2.2/intro/tutorial02/)

