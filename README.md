# tflow-dataset
A short tutorial repo to demonstrate efficient data pipeline and model training in tensorflow. To run training, install prerequistes and launch `train.py`

```
pip install -r requirements.txt
cd tflow_dataset 
python train.py
```

Output would will be something like:

```
{'person1': 'resources/lfw/Dalai_Lama/Dalai_Lama_0002.jpg', 'person2': 'resources/lfw/George_HW_Bush/George_HW_Bush_0006.jpg', 'same_person': False}
{'person1': 'resources/lfw/Dalai_Lama/Dalai_Lama_0002.jpg', 'person2': 'resources/lfw/George_HW_Bush/George_HW_Bush_0011.jpg', 'same_person': False}
step 0 log-loss 6.541984558105469
step 1 log-loss 11.30261516571045
...
step 98 log-loss 0.11421843618154526
step 99 log-loss 0.09954185783863068
Process finished with exit code 0
```
