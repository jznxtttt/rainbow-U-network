import numpy as np

class Manager(object):
    def __init__(self,permit,fee_rate):
        self.permit=permit
        self.fee_rate=fee_rate
        
        self.earn=0
        self.al_buy=0
        self.al_sell=0
    
    def reset(self):
        self.earn=0
        self.al_buy=0
        self.al_sell=0
    
    def total_money(self,mid):
        return (self.al_buy-self.al_sell)*mid+self.earn

    def buy_stock(self,mid):
        if self.al_buy<self.permit:
            self.al_buy=self.al_buy+1
            self.earn=self.earn-(1+self.fee_rate)*mid
    
    def sell_stock(self,mid):
        if self.al_sell<self.permit:
            self.al_sell=self.al_sell+1
            self.earn=self.earn+(1-self.fee_rate)*mid
    
    def close_position(self,mid):
        position=self.al_buy-self.al_sell
        if position>0:
            self.earn=self.earn+(1-self.fee_rate)*mid*position
            self.al_sell=self.al_sell+position
            
        if position<0:
            self.earn=self.earn-(1+self.fee_rate)*mid*(-position)
            self.al_buy=self.al_buy+(-position)



class TimingIntraday(object):
    def __init__(self,permit,fee_rate,date_list,stock_price_arr_list,\
                    state_arr_list,if_debug=True):
        self.fee_rate=fee_rate
        self.permit=permit
        self.if_debug=if_debug
        self.manager=Manager(self.permit,self.fee_rate)
        
        self.date_list=date_list
        self.stock_price_arr_list=stock_price_arr_list
        self.state_arr_list=state_arr_list

        self.total_day=len(self.stock_price_arr_list)
        if(self.if_debug):print("days in total:",self.total_day)
        if(self.if_debug):print("permit of unit:",self.permit)
        return

    def reset(self,day=None):
        self.manager.reset()
        self.current_time=0
        if day is None:
            day=np.random.randint(self.total_day)
        self.day=day
        self.stock_price_array=self.stock_price_arr_list[day]
        self.state_array=self.state_arr_list[day]
        self.total_step=len(self.state_array)
        return self.state_array[self.current_time], self.stock_price_array[self.current_time], self.manager.al_buy, self.manager.al_sell


    def step(self,action):
        price,next_price=self.stock_price_array[self.current_time:self.current_time+2]

        if self.current_time==self.total_step-2:
            orig_money=self.manager.total_money(price)
            self.manager.close_position(price)
            new_money=self.manager.total_money(next_price)
            r=new_money-orig_money
            
            s_= self.state_array[self.current_time+1]
            mid_ = next_price
            pb_ = 0
            ps_ = 0
            done = True

        else:
            orig_money=self.manager.total_money(price)
            if(action==1 and self.manager.al_buy<self.permit):
                self.manager.buy_stock(price)
            if(action==2 and self.manager.al_sell<self.permit):
                self.manager.sell_stock(price)
            new_money=self.manager.total_money(next_price)
            r=new_money-orig_money
            self.current_time+=1
            
            s_=self.state_array[self.current_time+1]
            mid_ = next_price
            pb_ = self.manager.al_buy
            ps_ = self.manager.al_sell
            done=False
            
        return s_, mid_, pb_, ps_, r, done, self.manager.total_money(price)