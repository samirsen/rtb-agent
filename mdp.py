import config
import pickle as pickle
import pandas as pd
import numpy as np

from utils import *
from scipy.optimize import linprog
from sklearn.metrics.cluster import normalized_mutual_info_score


def policy_optimize(reward, cost, budget, bins, max_bid, pctr_pdf):
    beq = [0 for i in range(0, bins + 1)]
    aeq = np.zeros((bins+1, bins*(max_bid + 1)))

    for i in range (bins):
        aeq[i][0:] = -pctr_pdf[i]
        aeq[i][i*(max_bid+1):(i+1)*(max_bid+1)]+=1

    aeq[bins][0:] = 1

    rho = linprog(-reward, A_ub=cost, b_ub=budget, A_eq=aeq, b_eq=beq, options={"maxiter": 5000, "tol": 1e-8})
    policy = [0 for i in range(0, bins)]
    for i in range(0, bins):
        policy[i]= np.argmax(rho.x[i * (max_bid+1):(i+1) * (max_bid+1)])
    return policy


def dp(ptcr, market_price_pdf, pctr_pdf, max_bid, bins, alpha, budget):
    actions = np.array(range(0, max_bid + 1))
    reward = np.zeros((bins, max_bid + 1))
    cost = np.zeros((bins, max_bid + 1))

    for i in range(bins):
        for j in range(max_bid + 1):
            k = j if j > 0 else 1
            reward[i][j] = sum(market_price_pdf[i][0:k]) * pctr[i]
            cost[i][j] = np.dot(market_price_pdf[i][0:k], actions[0:k)

    reward = reward.reshape((1, bins * (max_bid + 1))).ravel()
    cost = cost.reshape((1, bins * (max_bid + 1))).ravel()

    return policy_optimize(reward, cost, budget, bins, max_bid, pctr_pdf)


def predict(auction_in, policy, pctr, B, bid_log):
    log = "{:>10}\t{:>8}\t{:>8}\t{:>10}".format("bid_price", "win_price", "click", "budget")
    bid_log.write(log + "\n")

    b = B
    imp = 0
    clk = 0
    cost = 0
    cpmt = 0
    for i in range(auction_in.shape[0]):
        winprice = auction_in.iloc[i:i + 1, 1].values[0]
        click = auction_in.iloc[i:i + 1, 0].values[0]
        theta = auction_in.iloc[i:i + 1, 2].values[0]
        index = np.where(pctr <= theta)[-1][-1]
        a = max(0, min(b, policy[index]))

        if click == 1:
            cpmt += 1
        if a >= winprice:
            b -= winprice
            imp += 1
            clk += click
            cost += winprice

        log = "{:>10}\t{:>8}\t{:>8}\t{:>10}" .format(a, winprice, click, cost)
        bid_log.write(log + "\n")

    bid_log.flush()
    bid_log.close()
    return imp, clk, cost


def main():
    c0 = 1/32
    alpha = 1
    pctr_init = [0]
    bins = [10**(-7), 10**(-6), 10**(-5), 10**(-4), 10**(-3), 10**(-2), 10**(-1)]
    zip_bin = zip(bins, bins[1:])
    for i, j in zip_bin:
        pctr_bins = np.linspace(i, j, num=20)
        pctr_init = np.append(pctr_init, pctr_bins[:-1])
    pctr_bins2 = np.linspace(0.1, 1, num=10)
    pctr = np.append(pctr_init, pctr_bins2)
    # print (pctr)
    bins = len(pctr)-1

    src = "ipinyou"
    obj_type = "clk"
    clk_vp = 1
    train_file = 'train.theta.txt'

    camps = config.camps
    data_path = config.data_path
    max_market_price = config.max_market_price

    log_in = open(config.logPath + 'bid-stat' + "/{}_c0={}_obj={}_clkvp={}.txt".format(src, c0, obj_type, clk_vp), "w")
    print("logs in {}".format(log_in.name))
    log = "{:<60}\t {:>10}\t {:>8}\t {:>10}\t {:>8}\t {:>8}\t {:>9}\t {:>8}\t {:>8}"\
        .format("setting", "objective", "auction", "impression", "click", "cost", "win-rate", "CPM", "eCPC")
    log_in.write(log + "\n")

    for camp in camps:

        auction_in = pd.read_csv(data_path + camp + "/" + 'test.theta.txt', header=None, index_col=False, sep=' ',
                                 names=['click', 'winprice', 'pctr'])
        train = pd.read_csv(data_path + camp + "/" + train_file, header=None, index_col=False, sep=' ',
                            names=['click', 'winprice', 'pctr'])
        camp_info = pickle.load(open(data_path + camp + "/" + 'info.txt', "rb"))
        b = int(camp_info["cost_train"] / camp_info["imp_train"] * c0)
        m_pdf = calc_m_pdf(camp_info["price_counter_train"])
        max_bid = len(m_pdf) - 1

        # CMDP
        # Distritize pctr and get the conditional probability for market price: P(market_price| pctr)

        freq, pctr1 = np.histogram(train['pctr'], bins=pctr)
        price_range = [i for i in range(max_bid+2)]
        pctr_pdf = freq/(sum(freq))
        joint_pdf = np.histogram2d(train['pctr'], train['winprice'], bins=[pctr, price_range])
        cond_pdf = joint_pdf[0]/sum(sum(joint_pdf[0]))
        for i in range(bins):
            if pctr_pdf[i] == 0:
                cond_pdf[i][:] = m_pdf
            else:
                cond_pdf[i][:] = cond_pdf[i][:] / pctr_pdf[i]
        cond_pdf = np.nan_to_num(cond_pdf)
        m_pdf = cond_pdf

        # Log the bid price
        bid_log = open(
            config.logPath + "bid-log/{}/{}_camp={}_c0={}_obj={}.txt".format("CMDP", src, camp, c0, obj_type), "w")
        print("bid logs in {}".format(bid_log.name))

        b_total = int(camp_info["cost_train"] / camp_info["imp_train"] * c0 * auction_in.shape[0])
        print('Normalized Mutual Information Score is %.2f' % normalized_mutual_info_score(train['pctr'],
                                                                                           train['winprice']))

        policy = lr_cmdp(pctr, bins, m_pdf, pctr_pdf, max_bid, alpha, b)
        (imp, clk, cost) = run(auction_in, policy, pctr, b_total, bid_log)
        auction = auction_in.shape[0]
        win_rate = imp / auction * 100
        cpm = (cost / 1000) / imp * 1000
        ecpc = (cost / 1000) / clk
        # obj = clk
        obj = (auction_in['click'] == 1).sum()
        setting = "{}, camp={}, algo={}, c0={}".format(src, camp, "CMDP", c0)
        log = "{:<60}\t {:>10}\t {:>8}\t {:>10}\t {:>8}\t {:>8}\t {:>8.2f}%\t {:>8.2f}\t {:>8.2f}"\
            .format(setting, obj, auction, imp, clk, cost, win_rate, cpm, ecpc)
        print(log)
        log_in.write(log + "\n")
