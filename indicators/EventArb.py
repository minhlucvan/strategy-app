from numba import njit
import numpy as np
import vectorbt as vbt
import pandas as pd

# generate arbitrage signal
# input: stocks_df, events_df
# output: days_to_event represents the days to the nearest event < 0 for upcoming events and > 0 for past events
def generate_arbitrage_signal(stocks_df, events_df):    
    days_to_event_data = {}
    
    for stock in stocks_df.columns:
        stock_df = stocks_df[stock]
    
        if stock not in events_df.columns:
            days_to_event_data[stock] = [np.nan] * len(stock_df)
            continue
    
        event_df = events_df[stock]
        event_df = event_df.dropna()
        
        days_event_df = pd.DataFrame(index=stock_df.index, columns=['event_date','past_event_date', 'upccoming_event_date', 'days_from_event', 'days_to_event'])
        
        # check if there is any event
        if event_df.empty:
            days_to_event_data[stock] = [np.nan] * len(stock_df)
            continue
        
        event_df = pd.DataFrame(event_df)
        
        event_df['event_date'] = event_df.index
        
        event_df['event_date'] = pd.to_datetime(event_df['event_date'])
        
        days_to_event_df = event_df.reindex(stock_df.index, method='nearest')
        
        days_event_df['event_date'] = event_df['event_date']
        days_event_df['days_to_event'] = (days_to_event_df.index - days_to_event_df['event_date']).dt.days

        days_to_event_data[stock] = days_event_df['days_to_event']
        
    days_to_event_df = pd.DataFrame(days_to_event_data)
        
    return days_to_event_df


# apply function for EventArb
# entry signal: days_before_threshold days before the event
# exit signal: days_after_threshold days after the event
@njit
def apply_EventArb_nb(close, days_to_event, days_before_threshold, days_after_threshold):
  entries = np.full_like(close, False, dtype=np.bool_)
  exits = np.full_like(close, False, dtype=np.bool_)

  for i in range(close.shape[0]):
    for j in range(close.shape[1]):
      if days_to_event[i, j] >= -days_before_threshold and days_to_event[i, j] < 0:
        entries[i, j] = True
      if days_to_event[i, j] >= days_after_threshold and days_to_event[i, j] >= 0 and days_to_event[i, j] < days_after_threshold + 3:
        exits[i, j] = True

  return entries, exits

def get_EventArbInd():
    EventArb = vbt.IndicatorFactory(
        class_name="EventArb",
        input_names=["close", "days_to_event"],
        param_names=["days_before_threshold", "days_after_threshold"],
        output_names=["entries", "exits"]
    ).from_apply_func(apply_EventArb_nb)
    
    return EventArb
