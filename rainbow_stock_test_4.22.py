from stock_env.timing_intraday_new import TimingIntraday
from rainbow_stock_no_adjust2.rainbow_stock import DQNStockTradingAgent
import pandas as pd
import numpy as np
import os
import datetime
from tqdm import tqdm


PERMIT=10000
FEE_RATE_TRAIN=0.0003
FEE_RATE_TEST=0.0007
need_feature = pd.read_csv("need_feature.csv",index_col = 0)
need_feature = list(need_feature["0"].values) + ["mid"]
#need_feature=["2_1","2_5","1979_1","1979_5","616_1","616_5","average_price_diff5","high5","low5","average_price5","mid"]

drop_labels=["Unnamed: 0","SecurityID","PreClosePx","PxChange1",\
    	    "PxChange2","OpenPx","HighPx","LowPx","LastPx","NumTrades","TotalVolumeTrade",\
            "TotalValueTrade","totalofferqty","weightedavgofferpx","totalbidqty",\
            "weightedavgbidpx","UpLimitPx","DownLimitPx"]+["ask_price_"+str(i) for i in range(1,11)]+\
                ["bid_price_"+str(i) for i in range(1,11)]+["ask_vol_"+str(i) for i in range(1,11)]+\
                    ["bid_vol_"+str(i) for i in range(1,11)]+["rate"+str(i) for i in range(1,6)]+\
                        ["num_trade"+str(i) for i in range(1,6)]+["vol_trade"+str(i) for i in range(1,6)]+\
                        ["vol_trade"+str(i) for i in range(1,6)]+["value_trade"+str(i) for i in range(1,6)]+\
                        ["value_trade"+str(i) for i in range(1,6)]+["buy_value"+str(i) for i in range(1,6)]+\
                        ["buy_value"+str(i) for i in range(1,6)]+["sell_value"+str(i) for i in range(1,6)]
drop_feature = ["29_1","29_2","29_3","29_4","29_5"]


def select_time(time_arr):
    bool1=(time_arr.hour==9) & (time_arr.minute>=40)
    bool2=(time_arr.hour==10)
    bool3=(time_arr.hour==11) & (time_arr.minute<=25)
    bool4=(time_arr.hour==13) & (time_arr.minute>=10)
    bool5=(time_arr.hour==14) & (time_arr.minute<=55)
    return bool1 | bool2 | bool3 | bool4 | bool5

def multiply(df):
    for name in df.columns:
        if name[0].isdigit():
            df[name]=df[name]*100
    return df

# step 1: get df
train_df_list=[]
test_df_list=[]
train_date_list=[]
test_date_list=[]
file_path="D:\\feature3s\\"
print("reading files...")
for home,dirs,files in os.walk(file_path):
    for i_file in tqdm(files):
        month=int(i_file[1])
        date=i_file[:4]
        #if month!=1 and not (month==6 and int(date)%100==3):
        #    continue
        i_file=file_path+i_file
        df=pd.read_csv(i_file,index_col="SendTime")
        df.index = pd.to_datetime(df.index)
        df=df[select_time(df.index)]
        df=multiply(df)
        if len(df)==0:
            continue
        if month<=4:
            train_df_list.append(df)
            train_date_list.append(date)
        else:
            test_df_list.append(df)
            test_date_list.append(date)


# step 2: pre-processing 
print("select feature...")
for i in tqdm(range(len(train_df_list))):
    df = train_df_list[i]
    df = df.fillna(0)
    df = df.drop(labels = drop_feature+drop_labels,axis = 1)
    df = df[need_feature]
    train_df_list[i] = df

for i in tqdm(range(len(test_df_list))):
    df = test_df_list[i]
    df = df.fillna(0)
    df = df.drop(labels = drop_feature+drop_labels,axis = 1)
    df = df[need_feature]
    test_df_list[i] = df


print("normalization...")
train_df = pd.concat(train_df_list)
mean_dict = {}
std_dict = {}
for name in train_df.columns:
    mean = train_df[name].mean()
    std = train_df[name].std()
    mean_dict[name] = mean
    std_dict[name] = std

for i in tqdm(range(len(train_df_list))):
    df = train_df_list[i]
    for name in df.columns:
        if name == "mid":
            continue
        df[name] = (df[name]-mean_dict[name])/(std_dict[name]+ 1e-10)
        df.loc[df[name]>3,name] = 3
        df.loc[df[name]<-3,name] = -3
    train_df_list[i] = df

for i in tqdm(range(len(test_df_list))):
    df = test_df_list[i]
    for name in df.columns:
        if name == "mid":
            continue
        df[name] = (df[name]-mean_dict[name])/(std_dict[name]+ 1e-10)
        df.loc[df[name]>3,name] = 3
        df.loc[df[name]<-3,name] = -3
    test_df_list[i] = df

# step 3: create train env
stock_price_arr_list=[]
next_stock_price_arr_list=[]
state_arr_list=[]
next_state_arr_list=[]
for df in train_df_list:
    state_arr=df.drop(labels=["mid"],axis=1).values
    stock_price_arr=df["mid"].values
    state_arr_list.append(state_arr)
    stock_price_arr_list.append(stock_price_arr)

env=TimingIntraday(PERMIT,FEE_RATE_TRAIN,train_date_list,stock_price_arr_list,state_arr_list,False)

print("create test environment...")
stock_price_arr_list=[]
next_stock_price_arr_list=[]
state_arr_list=[]
next_state_arr_list=[]
for df in test_df_list:
    state_arr=df.drop(labels=["mid"],axis=1).values
    stock_price_arr=df["mid"].values
    state_arr_list.append(state_arr)
    stock_price_arr_list.append(stock_price_arr)

test_env=TimingIntraday(PERMIT,FEE_RATE_TEST,test_date_list,stock_price_arr_list,state_arr_list,False)

obs_dim=state_arr_list[0].shape[1]
act_dim = 3
env = env
test_env = test_env
memory_size = 4000*100
#memory_size = 4000*20
batch_size = 3000
target_update = 20
gamma = 0.99
v_min = -0.1
v_max = 0.1
atom_size = 51
n_step = 90
test_time = 100*10000
#test_time = 40*10000
train_total_time = 610*10000
is_from_load = False
is_to_save =True
dqn_load_name = "6000000_dqn_net_reload4_new_0.0002.pth"
dqn_target_load_name = "6000000_dqn_target_net_reload4_new_0.0002.pth"
dqn_save_name = "dqn_net_like_nn_low_gamma3_"+str(FEE_RATE_TRAIN)+".pth"
dqn_target_save_name = "dqn_target_net_like_nn_low_gamma3_"+str(FEE_RATE_TRAIN)+".pth"
is_need_test_in_train = False
is_need_test_in_test = True
#is_need_test_in_train = True
#is_need_test_in_test = False


time=datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d-%H-%M-%S')
save_folder = time+"rainbow_fig/"

os.mkdir(save_folder)
# step 4:learn

rainbow=DQNStockTradingAgent(
        obs_dim = obs_dim,
        act_dim = act_dim,
        env = env,
        test_env = test_env,
        memory_size = memory_size,
        batch_size = batch_size,
        target_update = target_update,
        gamma = gamma,
        # Categorical DQN parameters
        v_min = v_min,
        v_max = v_max,
        atom_size = atom_size,
        # N-step Learning
        n_step = n_step,
        test_time = test_time,
        save_folder = save_folder,
        is_from_load = is_from_load,
        is_to_save = is_to_save,
        dqn_load_name = dqn_load_name,
        dqn_save_name = dqn_save_name,
        dqn_target_load_name = dqn_target_load_name,
        dqn_target_save_name = dqn_target_save_name,
        is_need_test_in_train = is_need_test_in_train,
        is_need_test_in_test = is_need_test_in_test,
)

rainbow.train(train_total_time)