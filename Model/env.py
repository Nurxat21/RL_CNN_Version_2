import gym
import numpy as np
from numpy import random as rd

class StockTradingEnv(gym.Env):
    def __init__(
        self,
        config,
        initial_account=1e6,
        gamma=0.99,
        turbulence_thresh=99,
        min_stock_rate=0.1,
        max_stock=1e2,
        initial_capital=1e6,
        buy_cost_pct=1e-3,
        sell_cost_pct=1e-3,
        reward_scaling=2 ** -11,
        initial_stocks=None,
        cd2_dim=32,
    ):
        price_ary = config["price_array"]
        tech_ary = config["tech_array"]
        turbulence_ary = config["turbulence_array"]
        if_train = config["if_train"]
        self.price_ary = price_ary.astype(np.float32)
        self.tech_ary = tech_ary.astype(np.float32)
        self.turbulence_ary = turbulence_ary

        self.tech_ary = self.tech_ary * 2 ** -7
        self.turbulence_bool = (turbulence_ary > turbulence_thresh).astype(np.float32)
        self.turbulence_ary = (
            self.sigmoid_sign(turbulence_ary, turbulence_thresh) * 2 ** -5
        ).astype(np.float32)

        stock_dim = self.price_ary.shape[1]
        self.cd2_dim = cd2_dim
        self.gamma = gamma
        self.max_stock = max_stock
        self.min_stock_rate = min_stock_rate
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.initial_capital = initial_capital
        self.initial_stocks = (
            np.zeros(shape=(self.cd2_dim,stock_dim), dtype=np.float32)
            if initial_stocks is None
            else initial_stocks
        )

        # reset()
        self.day = None
        self.amount = None
        self.stocks = None
        self.total_asset = None
        self.gamma_reward = None
        self.initial_total_asset = None

        # environment information
        self.env_name = "StockEnv"
        # self.state_dim = 1 + 2 + 2 * stock_dim + self.tech_ary.shape[1]
        # # amount + (turbulence, turbulence_bool) + (price, stock) * stock_dim + tech_dim
        self.state_dim = (cd2_dim, (1 + 2 + 3 * stock_dim + self.tech_ary.shape[1]))
        # amount + (turbulence, turbulence_bool) + (price, stock) * stock_dim + tech_dim
        self.stocks_cd = None
        self.action_dim = stock_dim
        self.max_step = self.price_ary.shape[0] - 1
        self.if_train = if_train
        self.if_discrete = False
        self.target_return = 10.0
        self.episode_return = 0.0

        self.observation_space = gym.spaces.Box(
            low=-3000, high=3000, shape=(self.state_dim), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(self.action_dim,), dtype=np.float32
        )

    def reset(self):
        self.day = self.cd2_dim+1
        price = self.price_ary[self.day-self.cd2_dim: self.day]

        if self.if_train:
            self.stocks = (
                self.initial_stocks + rd.randint(0, 64, size=self.initial_stocks.shape)
            ).astype(np.float32)
            self.stocks_cool_down = np.zeros_like(self.stocks)
            self.amount = (
                self.initial_capital * rd.uniform(0.95, 1.05)
                - (self.stocks * price).sum(axis=1)
            )
        else:
            self.stocks = self.initial_stocks.astype(np.float32)
            self.stocks_cool_down = np.zeros_like(self.stocks)
            self.amount = self.initial_capital*np.ones(self.initial_stocks.shape[0])

        self.total_asset = self.amount + (self.stocks * price).sum(axis=1)
        self.initial_total_asset = self.total_asset
        self.gamma_reward = 0.0
        return self.get_state(price)  # state

    def step(self, actions):
        actions = (actions * self.max_stock).astype(int)

        self.day += 1
        price = self.price_ary[self.day-self.cd2_dim: self.day]
        self.stocks_cool_down[-1,:] += 1#self.stocks_cool_down += 1

        if self.turbulence_bool[self.day] == 0:
            min_action = int(self.max_stock * self.min_stock_rate)  # stock_cd
            for index in np.where(actions < -min_action)[0]:  # sell_index:
                if price[-1, index] > 0:  # Sell only if current asset is > 0###[-1,index]
                    sell_num_shares = min(self.stocks[-1, index], -actions[index])
                    self.stocks[-1, index] -= sell_num_shares
                    self.amount += (
                        price[-1, index] * sell_num_shares * (1 - self.sell_cost_pct)
                    )
                    self.stocks_cool_down[-1, index] = 0
            for index in np.where(actions > min_action)[0]:  # buy_index:
                if (
                    price[-1, index] > 0
                ):  # Buy only if the price is > 0 (no missing data in this particular date)
                    buy_num_shares = min(self.amount[-1] // price[-1, index], actions[index])
                    self.stocks[-1, index] += buy_num_shares
                    self.amount[-1] -= (
                        price[-1, index] * buy_num_shares * (1 + self.buy_cost_pct)
                    )
                    self.stocks_cool_down[-1, index] = 0

        else:  # sell all when turbulence
            self.amount += (self.stocks * price).sum(axis=1) * (1 - self.sell_cost_pct)
            self.stocks[-1,:] = 0
            self.stocks_cool_down[-1,:] = 0

        state = self.get_state(price)
        total_asset = self.amount + (self.stocks * price).sum(axis=1)
        reward = (total_asset - self.total_asset) * self.reward_scaling
        self.total_asset = total_asset

        self.gamma_reward = self.gamma_reward * self.gamma + reward
        done = self.day == self.max_step
        if done:
            reward = self.gamma_reward
            self.episode_return = total_asset / self.initial_total_asset

        return state, reward, done, dict()

    def get_state(self, price):
        amount = np.array(self.amount * (2 ** -12), dtype=np.float32)
        scale = np.array(2 ** -6, dtype=np.float32)
        output = np.column_stack(
            (
                amount,
                self.turbulence_ary[self.day-self.cd2_dim: self.day],
                self.turbulence_bool[self.day-self.cd2_dim: self.day],
                price * scale,
                self.stocks * scale,
                self.stocks_cool_down,
                self.tech_ary[self.day-self.cd2_dim: self.day],
            )
        )  # state.astype(np.float32)##hstack
        if output.shape[0]==self.cd2_dim:
            return output.reshape((1, output.shape[0], output.shape[1]))
        else:
            return output.reshape((output.shape[0], output.shape[1], output.shape[2],1))


    @staticmethod
    def sigmoid_sign(ary, thresh):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x * np.e)) - 0.5

        return sigmoid(ary / thresh) * thresh