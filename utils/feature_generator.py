import pickle
from utils import prompts, feature_processor
import json
import os
from utils.GPT import gpt as gpt
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm

# from IPython.display import clear_output
import numpy as np

from rouge_score import rouge_scorer

def calculate_rouge_l(hypothesis, reference):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    rouge_scores = scorer.score(hypothesis, reference)
    rouge_l_score = rouge_scores['rougeL'].fmeasure
    return rouge_l_score

class FeatureGenerator:
    def __init__(self, raw_feature_path, logger) -> None:
        self.fp = feature_processor.FeatureProcessor(raw_feature_path, logger)
        self.logger = logger
        open('features.jsonl', 'w').close()
        self.fp.load_features_from_str(json.dumps(prompts.groupy_by_stock_and_date_seed_features), filter=False, file=None)
        self.seeds = prompts.groupy_by_stock_seed_features
        self.seed_pool = []
        self.func_mut_pool = []
        self.const_mut_pool = []
        
    def get_mutation_format_tuple(self, mut_examples, mut_record, feature):
        def construct_example_str_single(record):
            return f'Example input: {json.dumps(record["feature"])}\nExample output: {chr(10).join([json.dumps(x) for x in record["new_features"]])}'
        def construct_example_str(records):
            return '\n'.join([construct_example_str_single(x) for x in records])
        
        if len(mut_record) < 8: return construct_example_str(mut_examples), 'None', json.dumps(feature)
        else: 
            rouge_l_score = [calculate_rouge_l(feature['expression'], record['feature']['expression']) for record in mut_record]
            candidate_records = sorted([mut_record[i] for i in [x[0] for x in sorted(enumerate(rouge_l_score), key=lambda x: x[1])]], key=lambda x: x['score'], reverse=True)
            return construct_example_str(candidate_records[:2]), construct_example_str(candidate_records[-2:]), json.dumps(feature)
    
    def func_mutation(self, features, mut_record=[], load_features=False):
        if not isinstance(features, list):
            features = [features]
        mut_features = []
        for feature in features:
            self.logger.info(f'func mutating {feature["new_feature_name"]}')
            prompt = prompts.func_mutation_prompt % self.get_mutation_format_tuple(prompts.func_mut_example, self.func_mut_pool, feature)
            resp = gpt(prompt)
            new_features = feature_processor.FeatureProcessor.extract_features(resp)
            cnt = len(new_features)
            for fe in new_features.copy():
                new_fe = dict()
                new_fe["new_feature_name"] = feature["new_feature_name"] + "_div_" + fe["new_feature_name"]
                # new_fe["expression"] = f'(data_stock_date["{feature["new_feature_name"]}"]) / (data_stock_date["{fe["new_feature_name"]}"])'
                new_fe["expression"] = f'({feature["expression"]}) / ({fe["expression"]})'
                new_fe["description"] = f'{feature["description"]} / {fe["description"]}'
                new_features.append(new_fe)
                cnt += 1
            if load_features: new_features = self.fp.load_features_from_json(new_features + [feature])
            self.logger.info(f'get {cnt} new features')
            mut_record.append({"feature": feature, "new_features": new_features})
            mut_features.extend([feature] + new_features)
            
        return mut_features
            
    
    def constant_mutation(self, features, mut_record=[], load_features=False):
        if not isinstance(features, list):
            features = [features]
        mut_features = []
        for feature in features:
            self.logger.info(f'constant mutating {feature["new_feature_name"]}')
            prompt = prompts.constant_mutation_prompt % self.get_mutation_format_tuple(prompts.const_mut_example, self.const_mut_pool, feature)
            resp = gpt(prompt)
            new_features = feature_processor.FeatureProcessor.extract_features(resp)
            if load_features: new_features = self.fp.load_features_from_json(new_features + [feature])
            cnt = len(new_features)
            self.logger.info(f'get {cnt} new features')
            mut_record.append({"feature": feature, "new_features": new_features})
            mut_features.extend([feature] + new_features)
        return mut_features
    
    def random_mutation(self, features, load_features=False):
        mut_features = []
        for feature in features:
            self.logger.info(f'random mutating {feature["new_feature_name"]}')
            prompt = prompts.random_mutation_prompt % json.dumps(feature)
            resp = gpt(prompt)
            new_features = feature_processor.FeatureProcessor.extract_features(resp)
            print(f'get {len(new_features)} new features')
            if load_features: new_features = self.fp.load_features_from_json(new_features + [feature])
            mut_features.extend([feature] + new_features)
        return mut_features
    
    def mutation(self, seed_features):
        mut_features = []
        for seed_feature in seed_features:
            func_mut_record = []
            const_mut_record = []
            func_mut_features = self.func_mutation(seed_feature, func_mut_record)
            const_mut_features = self.constant_mutation(func_mut_features, const_mut_record)
            new_features = self.fp.load_features_from_json(const_mut_features)
            mut_features.extend(new_features)
            score = self.fp.evaluate_ic([x["new_feature_name"] for x in mut_features])
            self.seed_pool.append({'seed': seed_feature, 'score': score})
            for i, record in tqdm(enumerate(func_mut_record), desc='func mutation evaluate'):
                record['score'] = self.fp.evaluate_ic([x["new_feature_name"] for x in record['new_features']])
                self.func_mut_pool.append(record)
            for i, record in tqdm(enumerate(const_mut_record), desc='const mutation evaluate'):
                record['score'] = self.fp.evaluate_ic([x["new_feature_name"] for x in record['new_features']])
                self.const_mut_pool.append(record)
        # clear_output()
        
        return mut_features
    
    def generate_new_seeds(self):
        self.logger.info('generating new seeds')
        
        if self.seed_pool:
            self.candidte_seeds = [
                x['seed'] for x in sorted(self.seed_pool, key=lambda x: x['score'], reverse=True)[:min(len(self.seed_pool), 6)]
            ]
            n = len(self.candidte_seeds)
            si = [[calculate_rouge_l(self.candidte_seeds[i]['expression'], self.candidte_seeds[j]['expression']) for i in range(n)] for j in range(n)]
            l = list(range(n))
            while len(l) > 3:
                index = [(i, j) for i in l for j in l if i != j]
                i, j = max(index, key=lambda x: si[x[0]][x[1]])
                l.remove(np.random.choice([i, j]))
            
            seed_use = [self.candidte_seeds[i] for i in l]
        else:
            seed_use = np.random.choice(prompts.groupy_by_stock_seed_features, 3)
        # seed_use = prompts.groupy_by_stock_seed_features[:3]
        prompt = prompts.groupy_by_stock_seed_prompt % '\n'.join([json.dumps(seed) for seed in seed_use])
        # prompt = prompts.generate_1_seed_prompt % '\n'.join([json.dumps(seed) for seed in seed_use])

        resp = gpt(prompt)
        # self.logger.info(resp)
        features = feature_processor.FeatureProcessor.extract_features(resp)
        self.logger.info(f'get {len(features)} new seeds')
        
        seed_pool_ = self.seed_pool.copy()
        cnt = 0
        for i, seed in enumerate(self.seed_pool):
            if seed['seed'] in self.candidte_seeds:
                seed_pool_.remove(seed)
                cnt += 1
            if cnt >= len(features): break
        
        return features
    
    def save(self, path = 'feature_generator_state.pkl'):
        with open(path, 'wb') as f:
            self.logger = None
            self.fp.logger = None
            pickle.dump(self, f)

    def run(self, init_seeds, quit_cnt=-1):
        if not self.seed_pool:
            self.mutation(init_seeds)
        
        while len(self.fp.stock_df['feature'].columns) < quit_cnt or quit_cnt==-1:
            seeds = self.generate_new_seeds()
            if not seeds: continue
            self.mutation(seeds)
    
    def gen_factors_with_inst(self, inst):
        resp = gpt(prompts.generate_seeds_with_user_instruction_prompt % ('\n'.join([json.dumps(x) for x in prompts.groupy_by_stock_seed_features]), inst))
        features = feature_processor.FeatureProcessor.extract_features(resp)
        self.logger.info(f'Generated {len(features)} seed features, mutating...')
        mut_features = self.mutation(features)
        self.logger.info(f'Generated {len(mut_features)} new features in total, saved in features.jsonl.')
        

# if __name__ == '__main__':
#     fg=FeatureGenerator('dataset/2008-01-01.2014-12-31_2015-01-01.2016-12-31_2017-01-01.2020-08-01_csi300/OHLCV')
#     print(fg.generate_new_seeds())