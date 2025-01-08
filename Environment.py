import gym
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding
import warnings
warnings.filterwarnings(action="ignore")

from Functions import Get_PosRet, Sharp_Ratio

class StockTradingEnv(gym.Env):
    def __init__(
        self, df, buy_cost_pct, sell_cost_pct, tech_indicator_list, seed, horizon=5):
        self.day = 0 # 交易第几天                             
        self.df = df # 交易数据                      
        self.buy_cost_pct = buy_cost_pct # 买入手续费         
        self.sell_cost_pct = sell_cost_pct # 卖出手续费       
        # self.state_space = state_space
        self.tech_indicator_list = tech_indicator_list # 技术指标(factor)名字的list
        self.action_space = spaces.Discrete(3)
        self.data = self.df.loc[self.day, :]
        self.terminal = False # 是否到达终点
        self.initial = True
        self.horizon = horizon
        self.state = self._initiate_state() # 从df中获取第0天的statement数据
        # initialize reward
        self.reward = 0
        # self.trades = 0
        self.signal_list = []
        self.episode = 0
        # memorize all the total balance change
        self.date_memory = []
        self._seed(seed=seed)

    def _buy_and_sell(self, actions):
        # action三种状态, [0]: sell, [1]: hold, [2]: buy
        # state是一个列表, [0]:仓位情况(1表示全仓做多, 0表示空仓), [1]:当日close价格, [1:tech_indicator+1]: 技术指标(特征工程)
        if self.state['hold'] == 1:
            # 当前持多仓，发出卖出信号
            if actions == 0:  
                self.state['hold'] = 0
                self.trades += 1
            # 其他情况说明持多仓,继续买入或者继续看多，仓位不变
            else:
                pass
        # 当前持空仓
        else:
            if actions == 2:  # 当前持空仓，发出买入信号
                self.state['hold'] = 1
                self.trades += 1
            else:
                pass

    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique()) - 1  # 需要确保index为日期,否则unique之后并不是所有日期,会有重复
        # actions = actions - 1  # actions is -1, 0, 1
        # print(f"now state: {self.state['hold']}; action: {actions}")
        if self.terminal:
            # 如果 one episode trade 结束
            # print(f"Episode: {self.episode} end;  total trades: {self.trades}")
            df = pd.DataFrame(self.date_memory, columns=['date'])
            df['signal'] = self.signal_list
            df['close'] = self.df.close
            df = Get_PosRet(df)
            sharp = Sharp_Ratio(df['return'])
            # print(f"Sharp Ratio: {sharp}")
            self.plot_df = df

        else:
            # 如果决策没有碰到数据集结束
            self.data = self.df.loc[self.day, :]
            self.date_memory.append(self._get_date())
            self.signal_list.append(actions)
            self._buy_and_sell(actions) # 更新持仓状态和交易记录
            self.day += 1
            self.next_state = self._update_state()

            # close_record = self.state[1]
            # 利用当日收盘的state和几日后的refer_state比较，计算reward
            if self.state['hold'] == 1:
                if actions == 0:  # 当前持多仓，发出卖出信号
                    self.reward = (1 - self.sell_cost_pct) * (2 - self.state['close_horizon'] / self.state['close']) -1
                else:
                    self.reward = self.state['close_horizon'] / self.state['close'] - 1
            else:
                if actions == 2:  # 当前持空仓，发出买入信号
                    self.reward = (1 - self.buy_cost_pct) * self.state['close_horizon'] / self.state['close'] - 1
                else:
                    self.reward = 1 - self.state['close_horizon'] / self.state['close']
        return self.state['indicators'], self.reward, self.terminal, self.next_state['indicators'] # return state, reward, 是否结束(Bool), info

    def reset(self):
        """
        初始化环境
        """
        self.state = self._initiate_state()
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.trades = 0
        self.reward = 0
        self.terminal = False
        self.date_memory = []
        self.signal_list = []
        self.episode += 1
        return self.state['indicators']

    def render(self):
        if self.state[0] > 0:
            print(f"交易日:{self.df.iloc[self.day]['date']}, 持仓: 全仓")
        else:
            print(f"交易日:{self.df.iloc[self.day]['date']}, 持仓: 空仓")
        return self.state

    def _initiate_state(self):
        state = {}
        state['hold'] = 0
        state['close'] = self.df.iloc[0, :]['close']
        state['close_horizon'] = self.df.iloc[0, :]['close_horizon']
        state['indicators'] = self.df.iloc[0, :][self.tech_indicator_list].to_list()
        return state

    def _update_state(self):
        state = {}
        state['hold'] = self.state['hold']
        state['close'] = self.df.iloc[self.day, :]['close']
        state['close_horizon'] = self.df.iloc[self.day, :]['close_horizon']
        state['indicators'] = self.df.iloc[self.day, :][self.tech_indicator_list].to_list()
        return state

    def _get_date(self):
        return self.data.date

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
