import sys
from pathlib import Path

parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)

from utils import feature_generator, prompts
from loguru import logger

raw_feature_path = 'datasets/2008-01-01.2014-12-31_2015-01-01.2016-12-31_2017-01-01.2020-08-01_csi300/OHLCV'
fg = feature_generator.FeatureGenerator(raw_feature_path, logger)

fg.fp.load_features_from_file('test.jsonl')