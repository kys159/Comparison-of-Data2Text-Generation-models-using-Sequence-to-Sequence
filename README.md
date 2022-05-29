# Data2Text-Generation

Liu et al.,(2018)'s model (Structure aware Seq2Seq) is structured in tensorflow1 and There is no version configured with **pytorch**. In addition, with the recent rapid growth of Graphic cards, it is necessary to release the code for the new version of tensorflow2 and pytorch using **CUDA11**.
This repository contains the contents of https://github.com/tyliupku/wiki2bio and contains code that was reimplemented using pytorch.

## 1. Requirements
~~~
CUDA 11.0
torch==1.7.1
nltk
rouge-score
pip install "numpy<1.17"
sklearn
~~~
## 2. Data
This study was conducted using WIKIBIO dataset. (https://github.com/tyliupku/wiki2bio)

![example](https://user-images.githubusercontent.com/61648914/170856147-916bac53-0ce7-4970-abb4-80dcce8fe186.png)

## 3. Paper
Comparison of Data2Text Generation models using Sequence-to-Sequence (Young-Seok Kim, Byung-Hun Lee, Kang Minji, Sung Won Han)

> In this study, we compare the modified encoding method and attention method using a sequence-to-sequence model in the Data2Text task. In addition, we propose a model that complements the shortcomings. Data2Text refers to the task of generating description from tabular data. The data consists of a set of values and fields, and there are significant performance differences depending on how the model learns these structures. Therefore, we compare two representative models utilizing sequence-to-sequence under the same conditions to find a more effective methodology for learning the structure of the data. In addition, it adds a copy mechanism to improve performance by allowing the output of words that are not in the word vocabulary, such as proper nouns. As a result of the experiment, encoding methods using values, fields, position information and dual attention, which combines values and field attention, showed the best performance. In addition, performance improvement in the sequence-to-sequence model was confirmed when the copy mechanism was utilized.

![figure_copy](https://user-images.githubusercontent.com/61648914/170856709-49182dff-e5da-4e25-b838-1eb8b9adc903.png)

## 4. Result

### BLEU-4 and Rouge-4-F1 for structure-aware seq2seq model(Liu et al., 2018) and Model with Copy mechanism 
![copy_score](https://user-images.githubusercontent.com/61648914/170856775-abd4de69-c41b-4d21-94af-d182ef1be49b.PNG)
