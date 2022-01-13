from Model.train_test import *
import matplotlib
from Model.env import StockTradingEnv
import Model.config as config
from Data.Data import *
from Data.Plot import backtest_stats, backtest_plot
import datetime
matplotlib.use('Agg')









env = StockTradingEnv
TRAIN_START_DATE = '2010-01-01'
TRAIN_END_DATE = '2021-01-01'

TEST_START_DATE = '2021-01-02'
TEST_END_DATE = '2022-01-01'
TECHNICAL_INDICATORS_LIST = ['macd',
                             'boll_ub',
                             'boll_lb',
                             'rsi_30',
                             'dx_30',
                             'close_30_sma',
                             'close_60_sma']

ERL_PARAMS = {"learning_rate": 3e-5,"batch_size": 520,"gamma":  0.998,
              "seed":311,"net_dimension":128, "target_step":5000, "eval_gap":60}

#demo for elegantrl
account_value_train = train(start_date = TRAIN_START_DATE,
                            end_date = TRAIN_END_DATE,
                            ticker_list = config.DOW_30_TICKER,
                            data_source = 'yahoofinance',
                            time_interval= '1D',
                            technical_indicator_list= TECHNICAL_INDICATORS_LIST,
                            drl_lib='elegantrl',
                            env=env,
                            model_name='ddpg',
                            cwd='./test_'+ 'ddpg',
                            erl_params=ERL_PARAMS,
                            break_step=5e5)
account_value_erl=test(start_date = TEST_START_DATE,
                        end_date = TEST_END_DATE,
                        ticker_list = config.DOW_30_TICKER,
                        data_source = 'yahoofinance',
                        time_interval= '1D',
                        technical_indicator_list= TECHNICAL_INDICATORS_LIST,
                        drl_lib='elegantrl',
                        env=env,
                        model_name='ddpg',
                        cwd='./test_'+'ddpg',
                        net_dimension = 128)
####Plot
baseline_df = DataProcessor('yahoofinance').download_data(ticker_list = ["^DJI"],
                                                            start_date = TEST_START_DATE,
                                                            end_date = TEST_END_DATE,
                                                            time_interval = "1D")
baseline_df = baseline_df.loc[33:,:]
stats = backtest_stats(baseline_df, value_col_name = 'close')
account_value_erl = pd.DataFrame({'date':baseline_df.date,'account_value':account_value_erl[0:len(account_value_erl)-1]})
print("==============Get Backtest Results===========")
now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')

perf_stats_all = backtest_stats(account_value=account_value_erl)
perf_stats_all = pd.DataFrame(perf_stats_all)
perf_stats_all.to_csv("./"+"/perf_stats_all_"+".csv.")
print("==============Compare to DJIA===========")
# S&P 500: ^GSPC
# Dow Jones Index: ^DJI
# NASDAQ 100: ^NDX
backtest_plot(account_value_erl,
             baseline_ticker = '^DJI',
             baseline_start = account_value_erl.loc[0,'date'],
             baseline_end = account_value_erl.loc[len(account_value_erl)-1,'date'])