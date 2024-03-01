from loguru import logger
import os
import argparse
from utils import feature_generator, prompts
import json, os, pickle

def main():
    parser = argparse.ArgumentParser(description="Process market and raw feature path")
    parser.add_argument("--raw_feature_path", type=str, default=r'datasets\2008-01-01.2014-12-31_2015-01-01.2016-12-31_2017-01-01.2020-08-01_csi300\OHLCV', help="Raw feature path")
    parser.add_argument("--n_alphas", type=int, default=300, help="Number of desired generated factors")
    args = parser.parse_args()

    logger.add('./logs/log_{time}', format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
    if not os.path.exists('./logs'): os.makedirs('./logs')
    open('./logs/chat_history.txt', 'w').close()

    raw_feature_path = args.raw_feature_path

    fg = feature_generator.FeatureGenerator(raw_feature_path, logger)
    fp = fg.fp

    fg.run(prompts.groupy_by_stock_seed_features, quit_cnt=args.n_alphas)

    print('best ic:', fp.best_ic)
    print('features:', len(fp.stock_df['feature'].columns))

if __name__ == '__main__':
    main()
