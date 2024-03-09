import warnings
import pandas as pd
import json
import numpy as np
import re
import os
from qlib.contrib.eva.alpha import calc_ic
from sklearn.linear_model import LinearRegression
from qlib.data.dataset.processor import CSRankNorm, RobustZScoreNorm, Fillna, DropnaLabel
from tqdm import tqdm
from types import NoneType

warnings.filterwarnings('ignore')

class FeatureProcessor:
    
    def __init__(self, raw_feature_path, logger) -> None:
        
        self.save_path = os.path.join(os.path.dirname(raw_feature_path), 'GPT')
        self.best_ic = 0
        
        train_df = pd.read_csv(os.path.join(raw_feature_path, 'train.csv'), index_col=[0, 1], header=[0, 1])
        valid_df = pd.read_csv(os.path.join(raw_feature_path, 'valid.csv'), index_col=[0, 1], header=[0, 1])
        test_df = pd.read_csv(os.path.join(raw_feature_path, 'test.csv'), index_col=[0, 1], header=[0, 1])
        
        self.valid_start_date = valid_df.index.get_level_values('datetime').min()
        self.test_start_date = test_df.index.get_level_values('datetime').min()
        
        self.proc_i = [RobustZScoreNorm(fit_start_time=train_df.index.get_level_values('datetime').min(), fit_end_time=train_df.index.get_level_values('datetime').max(), fields_group='feature', clip_outlier=True), Fillna(fields_group='feature')]
        self.proc_l = [CSRankNorm(fields_group='label'), DropnaLabel()]
        self.logger = logger
        
        train_data = train_df.copy()
        valid_data = valid_df.copy()
        test_data = test_df.copy()
        
        for proc in self.proc_i:
            if hasattr(proc, 'fit'): proc.fit(train_data)
            train_data = proc(train_data)
            valid_data = proc(valid_data)
            test_data = proc(test_data)
            
        for proc in self.proc_l:
            train_data = proc(train_data)
            valid_data = proc(valid_data)
        
        
        self.stock_df = pd.concat([train_data, valid_data, test_data])
        self.train_index = self.stock_df.index.get_level_values('datetime') < self.valid_start_date
        self.valid_index = (self.stock_df.index.get_level_values('datetime') >= self.valid_start_date) & (self.stock_df.index.get_level_values('datetime') < self.test_start_date)
        self.test_index = self.stock_df.index.get_level_values('datetime') >= self.test_start_date

        self.org_cols = self.stock_df.columns.to_list()[:-1]
        self.ic_tmp = self.evaluate_ic()
        self.stock_df_pool = pd.concat([train_df, valid_df, test_df], axis=0)['feature']
        self.stds = self.stock_df['feature'].std().values.tolist()
        # self.residual = self.stock_df['label'].values.copy().squeeze()
        
        self.feature_ic_effect_dic = dict()
        
    def exec_groupby_stock_date(self, expression:str):

        return eval(expression.replace('data_stock_date', 'self.stock_df_pool'))

    def exec_groupby_stock(self, expression:str, feature_name):
        grouped = self.stock_df_pool.groupby('instrument')
        data_stock_list = []
        for _, data_stock in tqdm(grouped, leave=False, desc=f'adding feature {feature_name}'):
            res = eval(expression)
            if isinstance(res, np.ndarray):
                res = pd.Series(res, index=data_stock.index)
            data_stock_list.append(res)
        return pd.concat(data_stock_list)

    def calc_ic_ric(self, pred, label, index):
        return calc_ic(pd.Series(pred, index=index), pd.Series(label, index=index))
    
    def pred_value(self, features=None, train_y=None, return_index=False):
        if isinstance(features, NoneType): cols = [c for c in self.stock_df.columns.tolist()]
        else: cols = [('feature', x) for x in features] + [('label', 'LABEL0')]
        train_data = self.stock_df[cols].loc[self.train_index | self.valid_index]
        test_data = self.stock_df[cols].loc[self.test_index]
        train_x = train_data['feature'].values
        if isinstance(train_y, NoneType): 
            train_y = train_data['label'].values.squeeze()
        model = LinearRegression(fit_intercept=True)
        model.fit(train_x, train_y)
        if not return_index: return model.predict(test_data['feature'].values)
        train_residual = train_y - model.predict(train_x)
        test_y = test_data['label'].values.squeeze()
        return model.predict(test_data['feature'].values), test_data.index, train_residual, test_y
    
    def evaluate_ic(self, features=None, valid=True):
        if isinstance(features, NoneType):
            cols = [c for c in self.stock_df.columns.tolist()]
        else:
            cols = [('feature', x) for x in features if x in self.stock_df['feature'].columns.tolist()] + [('label', 'LABEL0')]
        if len(cols) == 1: return 0
        # cols = self.stock_df.columns.tolist()
        train_data = self.stock_df[cols].loc[(self.train_index | self.valid_index) if not valid else self.train_index]
        test_data = self.stock_df[cols].loc[self.valid_index if valid else self.test_index]
        train_x = train_data['feature'].values
        train_y = train_data['label'].values.squeeze()
        model = LinearRegression(fit_intercept=True)
        model.fit(train_x, train_y)
        pred_y = model.predict(test_data['feature'].values)
        test_y = test_data['label'].values.squeeze()
        ic, ric = self.calc_ic_ric(pred_y, test_y, test_data.index)
        self.ric = ric.mean()
        self.icir = ic.mean() / ic.std()
        self.ricir = ric.mean() / ric.std()
        return ic.mean()
    
    def process_new_feature(self, feature_name):
        df = self.stock_df[[('feature', feature_name)]]
        df_train = df.loc[df.index.get_level_values('datetime') < self.test_start_date]
        df_test = df.loc[df.index.get_level_values('datetime') >= self.test_start_date]
        for proc in self.proc_i:
            if hasattr(proc, 'fit'): proc.fit(df_train)
            df_train = proc(df_train)
            df_test = proc(df_test)
        self.stock_df[('feature', feature_name)] = pd.concat([df_train, df_test], axis=0)[('feature', feature_name)]
    
    def add_new_feature(self, feature_name, feature, check_ic=False, process=True):
        self.stock_df_pool[feature_name] = feature.copy()
        self.stds.append(feature.std())
        self.stock_df[('feature', feature_name)] = feature.copy()

        if process: self.process_new_feature(feature_name)
        if check_ic:
            ic = self.evaluate_ic()
            self.feature_ic_effect_dic[feature_name] = ic - self.ic_tmp
            if ic < self.ic_tmp:
                del self.stock_df[('feature', feature_name)]
                raise ValueError(f'IC decrease from {self.ic_tmp:.6f} to {ic:.6f}')
            else:
                self.ic_tmp = ic
        self.stock_df = self.stock_df[sorted(self.stock_df.columns)]
    
    def filter_features(self, features=None, min_increase=0.0003, loop=False):
        if isinstance(features, NoneType):
            features = self.stock_df['feature'].columns
        features_except = [x for x in self.stock_df['feature'].columns if x not in features]
        y_pred, test_index, train_residual, test_y = self.pred_value(features_except, return_index=True)
        residual_pred = self.pred_value(features, train_y=train_residual)
        tmp_ic, _ = self.calc_ic_ric(y_pred + residual_pred, test_y, test_index)
        tmp_ic = np.nanmean(tmp_ic)
        while 1:
            # features = [('feature', x) for x in features if ('feature', x) in self.stock_df.columns]
            dropped = False
            features_tmp = features.copy()
            for feature in tqdm(features, desc='filter features', leave=False):
                features_tmp.remove(feature)
                residual_pred = self.pred_value(features_tmp, train_y=train_residual) if features_tmp else 0
                ic, _ = self.calc_ic_ric(y_pred + residual_pred, test_y, test_index)
                ic = np.nanmean(ic)

                if ic < tmp_ic + min_increase:
                    features_tmp.append(feature)
                else:
                    self.logger.info(f'drop feature {feature}, residual ic increase from {tmp_ic:.5f} to {ic:.5f}')
                    del self.stock_df[('feature', feature)]
                    tmp_ic = ic
                    dropped = True
            if not loop or not dropped: break
        test_ic = self.evaluate_ic(valid=False)
        self.logger.info(f'ic after filter {test_ic:.5f}')
        if test_ic > self.best_ic:
            self.best_ic = test_ic
            self.save(self.save_path)
        return features_tmp
    
    @staticmethod
    def extract_features(txt):
        if not '{' in txt: 
            return []
        else: 
            try:
                return list(map(json.loads, re.findall('\{.*?\}', txt)))
            except: return []
    
    def load_features_from_str(self, txt, **kwargs):
        features_tmp = self.extract_features(txt)
        if not features_tmp: 
            self.logger.info('nothing to add')
            return features_tmp
        self.load_features_from_json(features_tmp, **kwargs)
    
    def load_features_from_json(self, json_data, check_ic=False, filter=True, file='./features.jsonl', calc_ic=False):

        features_tmp = json_data.copy()
        i = 0
        added_features = []
        while i < len(features_tmp):
            feature = features_tmp[i]
            try: 
                feature_tuple = ('feature', feature['new_feature_name'])
                # if feature['new_feature_name'] in self.stock_df_pool.columns:
                #     raise ValueError('duplicate')
                if 'data_stock_date' in feature['expression']:
                    self.add_new_feature(feature['new_feature_name'], self.exec_groupby_stock_date(feature['expression']), check_ic)
                elif 'data_stock' in feature['expression']:
                    self.add_new_feature(feature['new_feature_name'], self.exec_groupby_stock(feature['expression'], feature['new_feature_name']), check_ic)
                std = self.stock_df[feature_tuple].std()
                # if std in self.stds:
                #     self.stock_df.drop(feature_tuple, axis=1, inplace=True)
                #     raise ValueError(f'duplicate with {self.stock_df_pool.columns.to_list()[self.stds.index(std)]}')
                self.logger.info(f'Successfully \tadd new feature {feature["new_feature_name"].ljust(30)}' + ('tmp ic: {self.ic_tmp:.6f}' if check_ic else ''))
                i += 1
                added_features.append(feature)
                if calc_ic: print(f'test ic: {self.evaluate_ic(valid=False)}')
            except Exception as e:
                try: self.logger.info(f'failed to \tadd new feature {feature["new_feature_name"].ljust(30)} info: {e}')
                except: print(f'fail: {e}')
                features_tmp.pop(i)
        
        if not added_features: return []
        
        if not check_ic: 
            self.ic_tmp = self.evaluate_ic()
            self.logger.info(f'ic: {self.evaluate_ic(valid=False)}')
        
        if filter:
            remain_features = self.filter_features([feature['new_feature_name'] for feature in added_features])
            added_features = [feature for feature in added_features if feature['new_feature_name'] in remain_features]
        
        if file:
            with open(file, 'a') as f:
                f.write('\n'.join([
                    json.dumps(feature) for feature in added_features
                ]) + '\n' if added_features else '')
        
        self.stock_df = self.stock_df[sorted(self.stock_df.columns.to_list())]
        
        return features_tmp

    def load_features_from_file(self, path, file=None, filter=False, **kwargs):
        with open(path) as f:
            txt = f.read().strip().replace('\n', ' ')
        return self.load_features_from_str(txt, file='' if path.endswith('features.jsonl') else './features.jsonl', filter=filter, **kwargs)
    
    def save(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        stock_df = self.stock_df
        stock_df[stock_df.index.get_level_values('datetime') < self.valid_start_date].to_csv(os.path.join(save_path, 'train.csv'))
        stock_df[(stock_df.index.get_level_values('datetime') >= self.valid_start_date) & (stock_df.index.get_level_values('datetime') < self.test_start_date)].to_csv(os.path.join(save_path, 'valid.csv'))
        stock_df[stock_df.index.get_level_values('datetime') >= self.test_start_date].to_csv(os.path.join(save_path, 'test.csv'))