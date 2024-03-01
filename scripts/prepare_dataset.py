import qlib
import pandas as pd
from qlib.constant import REG_CN
from qlib.utils import exists_qlib_data, init_instance_by_config
from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.data.handler import check_transform_proc

import os
import argparse
import warnings

def parse_arguments():
    parser = argparse.ArgumentParser(description='Specify time ranges for training, validation, and testing.')

    parser.add_argument('--train_start', type=str, default="2008-01-01", help='Start date for training data')
    parser.add_argument('--train_end', type=str, default="2014-12-31", help='End date for training data')

    parser.add_argument('--valid_start', type=str, default="2015-01-01", help='Start date for validation data')
    parser.add_argument('--valid_end', type=str, default="2016-12-31", help='End date for validation data')

    parser.add_argument('--test_start', type=str, default="2017-01-01", help='Start date for testing data')
    parser.add_argument('--test_end', type=str, default="2020-08-01", help='End date for testing data')
    parser.add_argument('--market', type=str, default="csi300", help='Market name')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    # use default data
    # NOTE: need to download data from remote: python scripts/get_data.py qlib_data_cn --target_dir ~/.qlib/qlib_data/cn_data
    provider_uri = "~/.qlib/qlib_data/crowd_data"  # target_dir
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    args = parse_arguments()
    
    market = args.market

    train_time = (args.train_start, args.train_end)
    valid_time = (args.valid_start, args.valid_end)
    test_time = (args.test_start, args.test_end)

    segments = {"train": train_time,"valid": valid_time,"test": test_time,}

    base_dir = './datasets'
    dataset_base_path = os.path.join(base_dir, '_'.join(['.'.join(x) for x in segments.values()])) + '_' + market
    if not os.path.exists(dataset_base_path): 
        os.makedirs(dataset_base_path)
    handler_name = 'OHLCV'

    data_handler_config = {
        "start_time": train_time[0],
        "end_time": test_time[1],
        "fit_start_time": train_time[0],
        "fit_end_time": train_time[1],
        "instruments": market,
        "infer_processors": [
            # {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
            # {"class": "Fillna", "kwargs": {"fields_group": "feature"}}
        ],
        "learn_processors": [
            # {"class": "DropnaLabel"},
            # {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}}
        ],
    }

    dataset_path = os.path.join(dataset_base_path, handler_name)
    if os.path.exists(os.path.join(dataset_path, "test.csv")):
        exit(0)
    if not os.path.exists(dataset_path): 
        os.makedirs(dataset_path)

    class OHLCV(DataHandlerLP):
        def __init__(
            self,
            instruments="csi500",
            start_time=None,
            end_time=None,
            freq="day",
            infer_processors=[],
            learn_processors=[],
            fit_start_time=None,
            fit_end_time=None,
            filter_pipe=None,
            inst_processors=None,
            **kwargs,
        ):
            infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
            learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

            data_loader = {
                "class": "QlibDataLoader",
                "kwargs": {
                    "config": {
                        "feature": self.get_feature_config(),
                        "label": kwargs.pop("label", self.get_label_config()),
                    },
                    "filter_pipe": filter_pipe,
                    "freq": freq,
                    "inst_processors": inst_processors,
                },
            }

            super().__init__(
                instruments=instruments,
                start_time=start_time,
                end_time=end_time,
                data_loader=data_loader,
                learn_processors=learn_processors,
                infer_processors=infer_processors,
                **kwargs,
            )

        def get_label_config(self):
            return ["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]

        @staticmethod
        def get_feature_config():
            fields = ['$open', '$high', '$low', '$close', '$volume', '$vwap']
            names = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'VWAP']

            return fields, names

    setattr(qlib.contrib.data.handler, handler_name, OHLCV)

    dataset_config = {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": handler_name,
                "module_path": "qlib.contrib.data.handler",
                "kwargs": data_handler_config,
            },
            "segments": segments
        },
    }

    dataset = init_instance_by_config(dataset_config)

    train_data = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    valid_data = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
    test_data = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
    train_data.to_csv(os.path.join(dataset_path, "train.csv"))
    valid_data.to_csv(os.path.join(dataset_path, "valid.csv"))
    test_data.to_csv(os.path.join(dataset_path, "test.csv"))