import holidays
from datetime import timedelta
import datetime
import numpy as np
import pandas as pd

def is_working_day(date, holidays_list):
    return date.weekday() < 5 and date not in holidays_list

def get_last_working_day_before(date, gap=1, max_iterations=30):
    if date is None or max_iterations == 0 or pd.isnull(date):
        return None

    if isinstance(date, (datetime.datetime, pd.Timestamp)):
        date = date.date()

    vietnam_holidays = holidays.VN()

    if isinstance(date, datetime.date):
        # Check for the number of working days to find
        working_days_found = 0
        while working_days_found < gap:
            date -= timedelta(days=1)
            if is_working_day(date, vietnam_holidays):
                working_days_found += 1
        
        date = pd.to_datetime(date)

        return date
    elif isinstance(date, np.datetime64):
        date = pd.to_datetime(date)
        date = date.to_pydatetime().date()
        
        # Check for the number of working days to find
        working_days_found = 0
        while working_days_found < gap:
            date -= timedelta(days=1)
            if is_working_day(date, vietnam_holidays):
                working_days_found += 1
                
        date = pd.to_datetime(date)
        return date
    else:
        print(f"Date {date} ({type(date)}) is not a datetime.date object.")
        return None
