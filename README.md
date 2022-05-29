# Data2Text-Generation

Liu et al.,(2018)'s model (Structure aware Seq2Seq) is structured in tensorflow1 and There is no version configured with pytorch. In addition, with the recent rapid growth of Graphic cards, it is necessary to release the code for the new version of tensorflow2 and pytorch using CUDA11.
This repository contains the contents of https://github.com/tyliupku/wiki2bio and contains code that was reimplemented using pytorch.

### 1. Requirements
~~~
CUDA 11.0
torch==1.7.1
nltk
rouge-score
pip install "numpy<1.17"
sklearn
~~~
### 2. Data
1) This study was conducted using WIKIBIO dataset. (https://github.com/tyliupku/wiki2bio)

![example](https://user-images.githubusercontent.com/61648914/170856147-916bac53-0ce7-4970-abb4-80dcce8fe186.png)

2) Preprocess
~~~
python mypreprocess.py
~~~
3) Main
~~~
python myMain.py
~~~

### #. Process : original code (forked from [tyliupku/wiki2bio](https://github.com/tyliupku/wiki2bio))
1) Original Data Download : RTV-Project 폴더 내에 다운받은 original_data 폴더 넣기  
https://drive.google.com/drive/folders/1kCTwTwEk_CE9nc6q0snXm3lALrz_ZUDN?usp=sharing 
2) Preprocess
~~~
python preprocess.py
~~~
3) Main
~~~
python Main.py
~~~
