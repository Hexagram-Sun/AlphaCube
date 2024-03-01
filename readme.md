# AlphaCube

## Introduction
This repo provides the source code of `AlphaCube`, an LLM-empowered alpha mining framework.

We employ ChatGPT to automate the process of exploring stock factors, enabling the efficient discovery of effective factors.

## Dataset
We run experiments on the crowd-source version of qlib data which can be downloaded by
```bash
wget https://github.com/chenditc/investment_data/releases/download/2023-06-01/qlib_bin.tar.gz
tar -zxvf qlib_bin.tar.gz -C ~/.qlib/qlib_data/crowd_data --strip-components=2
```

## Requirements

### Qlib

First, you should install the [qlib](https://github.com/microsoft/qlib) library.

```bash
pip install numpy
pip install --upgrade  cython
git clone https://github.com/microsoft/qlib.git && cd qlib
pip install .
```

### Packages
```bash
pip install -r requirements.txt
```

## Scripts

### Prepare dataset
```bash
python -u scripts/prepare_dataset.py \
--train_start "2008-01-01" --train_end "2014-12-31" \
--valid_start "2015-01-01" --valid_end "2016-12-31" \
--test_start "2017-01-01" --test_end "2020-08-01" \
--market "csi300"
```
The generated dataset will be in the **datasets** folder.

### Fill Openai API key
Replace the empty string after `openai.api_key` in **utils/GPT.py** with a valid API key.

### Generate stock Alphas
```bash
python main.py \
--raw_feature_path datasets/2008-01-01.2014-12-31_2015-01-01.2016-12-31_2017-01-01.2020-08-01_csi300/OHLCV \
--n_alphas 300
```

Arg `--n_alphas` is the number of desired generated factors. The generated alpha expressions will be in **features.jsonl**.