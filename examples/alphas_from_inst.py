import sys
from pathlib import Path

parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)

from utils import feature_generator, prompts
from loguru import logger

raw_feature_path = 'datasets/2008-01-01.2014-12-31_2015-01-01.2016-12-31_2017-01-01.2020-08-01_csi300/OHLCV'
fg = feature_generator.FeatureGenerator(raw_feature_path, logger)

prompt = 'I would like to exploit the volume-price divergence. A divergence occurs when price moves in one direction (upward or downward) while trading volume trends in the opposite direction. This divergence can signal potential reversals or continuation of trends. '

fg.gen_factors_with_inst(prompt)