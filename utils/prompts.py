groupy_by_stock_and_date_seed_prompt = '''If you are a seasoned data analyst and you need to extract stock factors, the stock data is provided in the format of data_stock_date: dict. Each key represents a factor. The data table currently includes the following factors:

OPEN: Opening price
CLOSE: Closing price
HIGH: Intraday high
LOW: Intraday low
VOLUME: Trading volume
VWAP: Volume Weighted Average Price within a given time period
%s
data_stock_date is a dict contains all features of a single stock on one day, the newly generated factors will be in the form of JSON dicts:

{"new_feature_name": ..., "expression": ..., "description": ...}

For example, you can generate factors like this:

%s

Please provide some COMPLEX features that you find meaningful. If you believe there are no new factors, please answer: "I believe there are no new factors."

Please directly provide all JSON code without any punctuation and DON'T answering in points, one line for each factor!
'''

groupy_by_stock_and_date_seed_features = [
    {"new_feature_name": "log_volume", "expression": "np.log(data_stock_date['VOLUME'] + 1)", "description": "log volume"},
    {"new_feature_name": "vwap_close_ratio", "expression": "data_stock_date['VWAP'] / data_stock_date['CLOSE']", "description": "vwap div close"},
    # {"new_feature_name": "price_range", "expression": "data_stock_date['HIGH'] - data_stock_date['LOW']", "description": "price range"},
    # {"new_feature_name": "price_range_percentage_change", "expression": "((data_stock_date['HIGH'] - data_stock_date['LOW']) / data_stock_date['OPEN'])", "description": "Percentage Change in Price Range"}
]

groupy_by_stock_seed_prompt = '''If you are a seasoned data analyst and you need to extract stock factors for next-day price prediction, the stock data is provided in the format of data_stock: pd.DataFrame, with a index of date. Each column represents a factor. The data table currently includes the following factors:

OPEN: Opening price
CLOSE: Closing price
HIGH: Intraday high
LOW: Intraday low
VOLUME: Trading volume
VWAP: Volume Weighted Average Price within a given time period
log_volume: log volume

the newly generated factors will be in the form of JSON dicts:

{"new_feature_name": ..., "expression": ..., "description": ...}

For example, you can generate factors like this:

%s

Please provide some complex features that you find meaningful. You can use rolling, quantile, max, min, rank, corr, or apply and lambda . DO NOT generate factors that are already included. you can use np.sign, np.log, etc. for unary operations. If you believe there are no new factors, please answer: "I believe there are no new factors."

Please directly provide all JSON code without any punctuation and DON'T answering in points, one line for each factor!
'''

generate_1_seed_prompt = '''If you are a seasoned data analyst and you need to extract stock factors for next-day price prediction, the stock data is provided in the format of data_stock: pd.DataFrame, with a index of date. Each column represents a factor. The data table currently includes the following factors:

OPEN: Opening price
CLOSE: Closing price
HIGH: Intraday high
LOW: Intraday low
VOLUME: Trading volume
VWAP: Volume Weighted Average Price within a given time period
log_volume: log volume

the newly generated factors will be in the form of JSON dicts:

{"new_feature_name": ..., "expression": ..., "description": ...}

For example, you can generate factors like this:

%s

Please provide only 1 feature that you find meaningful. You can use rolling, quantile, max, min, rank, corr, or apply and lambda . DO NOT generate factors that are already included. you can use np.sign, np.log, etc. for unary operations. If you believe there are no new factors, please answer: "I believe there are no new factors."

Please directly provide the JSON code without any punctuation.
'''

groupy_by_stock_seed_features = [
    
    # {"new_feature_name": "max_min_interval_60_days", "expression": "(data_stock['HIGH'].rolling(window=60, min_periods=1).apply(lambda x: x.argmax()) - data_stock['LOW'].rolling(window=60, min_periods=1).apply(lambda x: x.argmin())) / 60", "description": "The time period between previous lowest-price date occur after highest price date"},
    {"new_feature_name": "idmax_60_days", "expression": "data_stock['CLOSE'].rolling(window=60).apply(lambda x: ((len(x) - 1) - x.argmax()) / len(x), raw=True)", "description": "The number of days between current date and previous highest price date in recent 60 days"},
    {"new_feature_name": "close_log_volume_change_corr_60_days", "expression": "data_stock['CLOSE'].pct_change().rolling(window=60, min_periods=1).corr(data_stock['log_volume'].pct_change())", "description": "close change corr log volume change"},
    {"new_feature_name": "gain_div_change_60_days", "expression": "(data_stock['CLOSE'] - data_stock['CLOSE'].shift(1)).clip(lower=0).rolling(window=5).sum() / ((data_stock['CLOSE'] - data_stock['CLOSE'].shift(1)).abs().rolling(window=5).sum() + 1e-12)", "description": "The total gain / the absolute total price changed"}, 
    {"new_feature_name": "80th_percentile_60_days", "expression": "data_stock['CLOSE'].rolling(window=60).quantile(0.8)", "description": "Quantile 80 of 60 days"},
    {"new_feature_name": "price_position_60_days", "expression": "(data_stock['CLOSE'] - data_stock['LOW'].rolling(window=60, min_periods=1).min())/(data_stock['HIGH'].rolling(window=60, min_periods=1).max() - data_stock['LOW'].rolling(window=60, min_periods=1).min())", "description":"Represent the price position between upper and lower resistent price for past 60 days."},
    {"new_feature_name": "low_price_60_days", "expression": "data_stock['LOW'].rolling(window=60, min_periods=1).min() / data_stock['CLOSE']", "description":"The low price for past 60 days"},
    {"new_feature_name": "go_down_percentage_60_days", "expression": "data_stock['CLOSE'].diff(1).rolling(window=60, min_periods=1).apply(lambda x: (x < 0).mean())", "description":"The percentage of days in past 60 days that price go down"},    

]

func_mutation_prompt = '''If you are a seasoned data analyst and you need to extract stock factors for next-day price prediction, the stock data is provided in the format of data_stock: pd.DataFrame, with a index of date. Each column represents a factor. The data table currently includes the following factors:

OPEN: Opening price
CLOSE: Closing price
HIGH: Intraday high
LOW: Intraday low
VOLUME: Trading volume
VWAP: Volume Weighted Average Price within a given time period

the newly generated factors will be in the form of JSON dicts:

{"new_feature_name": ..., "expression": ..., "description": ...}

Given a factor, perform a mutation operation on it. You can replace a function, a factor, or other single mutation operation in the expression, and change new_feature_name and description correspondingly. DO NOT change the constant in the expression. Provide the mutated factor directly.

Here are some effective mutation examples:
%s

Here are some ineffective mutation examples:
%s

Directly give the mutated output(s). 

Input: %s
Output: 
'''

random_mutation_prompt = '''If you are a seasoned data analyst and you need to extract stock factors for next-day price prediction, the stock data is provided in the format of data_stock: pd.DataFrame, with a index of date. Each column represents a factor. The data table currently includes the following factors:

OPEN: Opening price
CLOSE: Closing price
HIGH: Intraday high
LOW: Intraday low
VOLUME: Trading volume
VWAP: Volume Weighted Average Price within a given time period

the newly generated factors will be in the form of JSON dicts:

{"new_feature_name": ..., "expression": ..., "description": ...}

Given a factor, perform a mutation operation on it. You can replace a function, a factor, or other single mutation operation in the expression, and change new_feature_name and description correspondingly, you can also change the constant in the expression. Provide the mutated factor directly.

Here is a mutation examples:
Example input: {"new_feature_name": "go_down_percentage_30_days", "expression": "data_stock['CLOSE'].diff(1).rolling(window=30, min_periods=1).apply(lambda x: (x < 0).mean())", "description":"The percentage of days in past 30 days that price go down"}
Example output: {"new_feature_name": "go_up_percentage_30_days", "expression": "data_stock['CLOSE'].diff(1).rolling(window=30, min_periods=1).apply(lambda x: (x > 0).mean())", "description":"The percentage of days in past 30 days that price go up"}]}
{"new_feature_name": "go_down_percentage_10_days", "expression": "data_stock['CLOSE'].diff(1).rolling(window=10, min_periods=1).apply(lambda x: (x < 0).mean())", "description":"The percentage of days in past 10 days that price go down"}
{"new_feature_name": "go_down_percentage_5_days", "expression": "data_stock['CLOSE'].diff(1).rolling(window=5, min_periods=1).apply(lambda x: (x < 0).mean())", "description":"The percentage of days in past 5 days that price go down"}


Directly give some mutated outputs. 

Input: %s
Output: 
'''

constant_mutation_prompt = '''If you are a seasoned data analyst and you need to extract stock factors for next-day price prediction, the stock data is provided in the format of data_stock: pd.DataFrame, with a index of date. Each column represents a factor. The data table currently includes the following factors:

OPEN: Opening price
CLOSE: Closing price
HIGH: Intraday high
LOW: Intraday low
VOLUME: Trading volume
VWAP: Volume Weighted Average Price within a given time period

the newly generated factors will be in the form of JSON dicts:

{"new_feature_name": ..., "expression": ..., "description": ...}

Given a factor, change the constant in the expression, and change the new_feature_name and description correspondingly, to get several different outputs.

Here are some effective mutation examples:
%s

Here are some ineffective mutation examples:
%s


Directly give the mutated outputs. DO NOT use the operation that pandas does not support(like idxmin, use argmin instead).

Input: %s
Output: 
'''

func_mut_example = [{"feature": {"new_feature_name": "go_down_percentage_30_days", "expression": "data_stock['CLOSE'].diff(1).rolling(window=30, min_periods=1).apply(lambda x: (x < 0).mean())", "description":"The percentage of days in past 30 days that price go down"}, "new_features": [{"new_feature_name": "go_up_percentage_30_days", "expression": "data_stock['CLOSE'].diff(1).rolling(window=30, min_periods=1).apply(lambda x: (x > 0).mean())", "description":"The percentage of days in past 30 days that price go up"}]}, {"feature": {"new_feature_name": "low_price_30_days", "expression": "data_stock['LOW'].rolling(window=30, min_periods=1).min() / data_stock['CLOSE']", "description":"The low price for past 30 days"}, "new_features": [{"new_feature_name": "high_price_30_days", "expression": "data_stock['HIGH'].rolling(window=30, min_periods=1).max() / data_stock['CLOSE']", "description":"The high price for past 30 days"}]}]

const_mut_example = [{"feature": {"new_feature_name": "max_min_interval_30_days", "expression": "(data_stock['HIGH'].rolling(window=30, min_periods=1).apply(lambda x: x.argmax()) - data_stock['LOW'].rolling(window=30, min_periods=1).apply(lambda x: x.argmin())) / 30", "description": "The time period between previous lowest-price date occur after highest price date"}, "new_features": [{"new_feature_name": "max_min_interval_5_days", "expression": "(data_stock['HIGH'].rolling(window=5, min_periods=1).apply(lambda x: x.argmax()) - data_stock['LOW'].rolling(window=5, min_periods=1).apply(lambda x: x.argmin())) / 5", "description": "The time period between previous lowest-price date occur after highest price date in recent 5 days"},
{"new_feature_name": "max_min_interval_10_days", "expression": "(data_stock['HIGH'].rolling(window=10, min_periods=1).apply(lambda x: x.argmax()) - data_stock['LOW'].rolling(window=10, min_periods=1).apply(lambda x: x.argmin())) / 10", "description": "The time period between previous lowest-price date occur after highest price date"},
{"new_feature_name": "max_min_interval_20_days", "expression": "(data_stock['HIGH'].rolling(window=20, min_periods=1).apply(lambda x: x.argmax()) - data_stock['LOW'].rolling(window=20, min_periods=1).apply(lambda x: x.argmin())) / 20", "description": "The time period between previous lowest-price date occur after highest price date"},
{"new_feature_name": "max_min_interval_60_days", "expression": "(data_stock['HIGH'].rolling(window=60, min_periods=1).apply(lambda x: x.argmax()) - data_stock['LOW'].rolling(window=60, min_periods=1).apply(lambda x: x.argmin())) / 60", "description": "The time period between previous lowest-price date occur after highest price date"}]}]