import pandas as pd
import matplotlib.pylab as plt
import numpy as np
import copy

# 根据信号更新持仓和投资金额
def Get_PosRet(df):
    # df['signal_lag1'] = df['signal'].shift(1).fillna(0)
    df['is_position'] = 0
    df['investment'] = 0
    index = df.index
    for i in range(len(index)):
        bar = df.loc[index[i], :] # 在这里定义一个bar是为了偷懒，反复写df.loc太麻烦了......
        if bar.is_position == 0:
            if bar.signal == 2:
                df.loc[index[i], "investment"] = bar.close
                df.loc[i:, "is_position"] = 1
            else:
                df.loc[index[i], "investment"] = df.loc[index[i-1], "investment"] # 向前追溯上一次的买价
        else: # 当前满仓 
            if bar.signal == 0:
                df.loc[index[i], "investment"] = bar.close
                df.loc[i:, "is_position"] = 0
            else:
                df.loc[index[i], "investment"] = df.loc[index[i-1], "investment"] # 向前追溯上一次的买价
    df['return'] = df['investment'].pct_change()
    df['return'] = df['return'].replace(np.inf, np.nan).fillna(0)
    df.loc[df['signal']==2, "return"] = 0
    return df

def Sharp_Ratio(rt):
    annualized_return = (1 + rt.mean()) ** 252 - 1
    annualized_volatility = rt.std() * (252 ** .5)
    return annualized_return / annualized_volatility

def Plot_fig(df):
    plot_df = copy.deepcopy(df)
    plot_df.set_index("date", inplace=True)

    # %matplotlib inline
    plt.figure(facecolor="white", figsize=(9,4))
    # trade signal 
    pre_signal = None
    for i in range(len(plot_df)):
        if plot_df['signal'][i] != pre_signal:
            if plot_df['signal'][i] == 0:
                plt.plot(plot_df.index[i], plot_df['close'][i], "v", color="grey", alpha= .7)
            elif plot_df['signal'][i] == 2:
                plt.plot(plot_df.index[i], plot_df['close'][i], "^", color="purple", alpha= .7)
            pre_signal =  plot_df['signal'][i]
    # Trade curve
    non_zero_index = plot_df[plot_df['investment'] != 0].index[0]
    plt.plot(plot_df['close'], linewidth=2, label="Market price")
    plt.plot(plot_df['investment'][non_zero_index:], label="Holding value")
    # show
    plt.legend()
    plt.show()


def MDD(df: pd.DataFrame):
    md = ((df.cummax() - df) / df.cummax()).max()
    return md

def InfoRatio(df, benchmark):
    ex_return = df - benchmark
    information = np.sqrt(len(ex_return)) * ex_return.mean() / ex_return.std()
    return information

