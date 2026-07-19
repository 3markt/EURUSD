select * from forex_raw fr order by date_time desc;

delete from forex_raw where date_time = '2023-04 21:05';

insert into forex_raw (currency, date_time, cnt_diff_vals, "open", "close", low, high)
select currency, '2023-04-02 21:10', cnt_diff_vals, "open", "close", low, high
from forex_raw
where date_time = '2023-03-31 21:00';