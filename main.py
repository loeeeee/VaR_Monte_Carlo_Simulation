import random
import time

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from scipy.stats import norm

def INIT_SETTINGS(): # this is basically all the things you need to change to play with the program
    global MODE
    global confidence
    global MC_n
    global MC_t
    global plot_flag
    global rows
    global MC_test_n
    global MC_test_flag

    MODE=0 #MODE 0 is testing the [rows] amount of rows. MODE 1 is testing the rest of the rows. MODE 2 is testing all rows. Default to 0.
    confidence = .95 #Default to 0.95
    rows=1000 #Number of rows for VaR calculation, rest of the row are for testing. MODE 2 use this in the method "compare two parts of the data". Default to 1000
    plot_flag=True #To plot or not to. Default to True.

    MC_n=100 #Number of samples in each Monte Carlo Method testing. Default to 100.
    MC_t=1000 #Number of testing in Monte Caelo simulation method. Default to 1000.

    MC_test_flag=False #To run the MC_test or not MC_test is to test the consistenct of the Monte Carlo simulation method. Default to False.
    MC_test_n=100 #Number of testing the consistency of the method. Default to 100.
    return

def INIT():
    INIT_SETTINGS()
    READ()

    global delta_price
    global delta_profit
    global num_obs_in_tail
    global num_days
    global tail_risk

    delta_price=np.array(close_price-open_price)
    delta_profit = delta_price / open_price
    num_days = len(delta_price)
    tail_risk = 1 - confidence
    num_obs_in_tail = round(tail_risk * num_days)

    print("Number of days: {}\nConfidence Level: {}\nNumber of samples in the tail: {}". \
          format(num_days, confidence, num_obs_in_tail))
    print()
    return

def READ():
    global df
    global close_price
    global open_price

    if(MODE==0):
        df = pd.read_csv("GOOGL.csv", nrows=rows)

        close_price = df['Close']
        open_price = df['Open']
    elif(MODE==1):
        df = pd.read_csv("GOOGL.csv", skiprows=range(1,rows))

        close_price = df['Close']
        open_price = df['Open']
    elif(MODE==2):
        df = pd.read_csv("GOOGL.csv")

        close_price = df['Close']
        open_price = df['Open']

        COMPARE_TWO_PART()
    else:
        print("ERROR MODE NUMBER!")
    return
def COMPARE_TWO_PART():
    if(plot_flag==True):
        df_1 = pd.read_csv("GOOGL.csv", nrows=rows)
        close_price_1 = df_1['Close']
        open_price_1 = df_1['Open']
        df_2 = pd.read_csv("GOOGL.csv", skiprows=range(1, rows))
        close_price_2 = df_2['Close']
        open_price_2 = df_2['Open']

        delta_price_1 = np.array(close_price_1 - open_price_1)
        delta_profit_1 = delta_price_1 / open_price_1
        mean_1 = np.mean(delta_profit_1)
        std_1 = np.std(delta_profit_1)

        delta_price_2 = np.array(close_price_2 - open_price_2)
        delta_profit_2 = delta_price_2 / open_price_2
        mean_2 = np.mean(delta_profit_1)
        std_2 = np.std(delta_profit_1)

        COUNT,BINS,IGNORED= plt.hist(delta_profit_1,bins=100,color="orange",ec="orange",alpha=0.5)
        plt.plot(BINS, 1 / (std_1 * np.sqrt(2 * np.pi)) *
                 np.exp(- (BINS - mean_1) ** 2 / (2 * std_1 ** 2)),
                 linewidth=2,color="orange" )
        COUNT,BINS,IGNORED= plt.hist(delta_profit_2,bins=100,color="skyblue",ec="skyblue",alpha=0.5)
        plt.plot(BINS, 1 / (std_2 * np.sqrt(2 * np.pi)) *
                 np.exp(- (BINS - mean_2) ** 2 / (2 * std_2 ** 2)),
                 linewidth=2, color="skyblue")
        plt.xlabel('Return rate')
        plt.ylabel('Number of days')
        plt.show()
    return

def PLOT_DATA():
    if(plot_flag==True):
        CLOSE_PRICE()
        CLOSE_PRICE_HIST()
        RETURN_RATE()
        RETURN_RATE_HIST()
    return

def CLOSE_PRICE():
    plt.plot(np.arange(len(close_price)), close_price, color = "orange", alpha =0.7)
    plt.xlabel('Number of days')
    plt.ylabel('Close price')
    plt.show()
    return

def CLOSE_PRICE_HIST():
    plt.hist(close_price, bins=100, color = "orange", alpha =0.7,ec = 'orange')
    plt.xlabel('Closing price')
    plt.ylabel('Number of days')
    plt.show()
    return

def RETURN_RATE():
    plt.plot(np.arange(len(close_price)), delta_profit, color = "orange", alpha =0.7)
    plt.xlabel('Number of days')
    plt.ylabel('Return rate')
    plt.show()
    return

def RETURN_RATE_HIST():
    plt.hist(delta_profit, bins=100, color = "orange", alpha =0.7, ec='orange')
    plt.xlabel('Return rate')
    plt.ylabel('Number of days')
    plt.show()
    return

def HM():
    print("Historical Method:")

    start=time.time()

    sorted_daily_profit_loss = np.sort(delta_profit)
    tailed_sorted_daily_profit_loss = sorted_daily_profit_loss[:num_obs_in_tail]
    hs_var = sorted_daily_profit_loss[num_obs_in_tail]

    stop=time.time()
    eve_execution = (stop - start)

    print("Performence: %f ms"%eve_execution)
    print("{}% VaR = {}". \
          format(confidence * 100, hs_var))
    print()

    if(plot_flag==True):
        plt.hist(sorted_daily_profit_loss[num_obs_in_tail:], bins=100,color='skyblue',ec='skyblue',alpha=0.7)
        plt.hist(tailed_sorted_daily_profit_loss, bins=120, color="orange", ec="orange",alpha=0.7)
        plt.xlabel('Return rate')
        plt.ylabel('Number of days')
        plt.show()

    return

def VC():
    print("Variance-Covariance Method:")

    start=time.time()

    mean = np.mean(delta_profit)
    std = np.std(delta_profit)
    delta_var = norm.ppf(tail_risk, loc=mean, scale=std)

    stop=time.time()
    eve_execution = (stop - start)

    print("Performence: %f ms"%eve_execution)
    print("Mean:{}\nStandard Deviation:{}".format(mean, std))
    print("{}% VaR = {}".\
    format(confidence * 100, delta_var))
    print()

    if(plot_flag==True):
        COUNT,BINS,IGNORED= plt.hist(delta_profit,bins=100,alpha=0.7,color="orange",ec="orange")
        plt.plot(BINS,1/(std * np.sqrt(2 * np.pi)) *
                   np.exp( - (BINS - mean)**2 / (2 * std**2) ),
             linewidth=2,)
        plt.xlabel('Return rate')
        plt.ylabel('Number of days')
        plt.show()

    return

def MC(output= True):
    if(output==True):
        print("Monte Carlo Simulation Method:")
        start = time.time()

    VaR=[]
    for k in range(MC_t):
        picked_data = []
        for i in range(MC_n):
            picked_data.append(delta_profit[random.randint(0,len(delta_profit)-1)])
        mean = np.mean(picked_data)
        std = np.std(picked_data)
        VaR.append(norm.ppf(tail_risk,loc=mean, scale=std))
    rVaR=np.average(VaR)

    if(output==True):
        stop = time.time()
        eve_execution = (stop - start)

        print("Performence: %f ms" % eve_execution)
        print("{}% VaR = {}".format(confidence * 100, rVaR))
        print()

    return rVaR

def MC_TEST():
    if(MC_test_flag==False):
        return

    print("Testing the Consistency of Monte Carlo Simulation:")
    print("Test parameter:\n\tNumber of sampling: %d\n\tNumber of testing: %d\n\tNumber of consistency testing: %d"%(MC_n,MC_t,MC_test_n))

    start=time.time()

    res=[]
    for i in range(MC_test_n):
        res.append(MC(output=False))
        print("Working on it... %f%%" % (i/ MC_test_n*100)) #feel free to disable this line #this line can show if your system hangs
    avg=np.average(res)
    std=np.std(res)

    stop=time.time()
    eve_execution=(stop-start)/MC_test_n

    print("Performence: %f ms"%eve_execution)
    print("Average: %f"%avg)
    print("Standard deviation: %f"%std)
    return

def main():
    INIT()

    PLOT_DATA()

    HM()
    VC()
    MC()

    MC_TEST()

    return

if __name__ == '__main__':
    main()