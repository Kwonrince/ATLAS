# ATLAS : Adapting Triplet Loss to Abstractive Summarization
https://youtu.be/KqJ1miFdxvw


## Framework
<img src="https://user-images.githubusercontent.com/72617445/171579714-cfc80cef-568f-407c-8aac-19735d6aaa57.png" width="60%" height="40%">


## Dependecies
- python >= 3.7
- pytorch >= 1.10
- transformers == 4.19.2
- nltk
- tqdm
- numpy
- pandas

## Data preprocess
CNN/DM dataset will be automatically downloaded and make positive & negative samples.

You can change ratio.
```python
python preprocess.py
```

## Train
You can check the details in the parser.

- Train baseline
```python
python main.py --triplet False --batch_size 4 --devices 0_1_2_3
```

- Train ATLAS
```python
python main.py --batch_size 4 --devices 0_1_2_3
```

## Evaluate
Check inference.py
