### 1. Installation

If you don't have pipenv:
```
pip3 install pipenv
```

After installing pipenv:
```
pipenv install
```

### 2. (To use in jupyter notebook) Run

```
pipenv run jupyter notebook
```

### 3. (To use in VS code) Run
```
pipenv shell
```

Reference:
https://machinelearningmastery.com/prepare-text-data-machine-learning-scikit-learn/
https://medium.com/swlh/a-simple-guide-on-using-bert-for-text-classification-bbf041ac8d04
https://www.tensorflow.org/text/tutorials/classify_text_with_bert

## To run in GPU farm

### 1. Connect to HKU vpn
cisco client: vpn2fa.hku.hk
login with hku portal username and password and mail 2FA

### 2. SSH to GPU farm gateway
```
ssh -X uid@gpu2gate1.cs.hku.hk
```

### 3. Launch GPU node and jupyter
```
gpu-interactive
conda activate tensorflow_2_11
jupyter-lab --no-browser --FileContentsManager.delete_to_trash=False
```

### 4. Port forward GPU node via SSH
```
hostname -i
```
New terminal:
```
ssh -L 8888:localhost:8888 <your_gpu_acct_username>@10.XXX.XXX.XXX
```

### 5. Go to jupyer with token
http://localhost:8888/lab?token=xxxxxxxxx

More info:
https://www.cs.hku.hk/gpu-farm/home
