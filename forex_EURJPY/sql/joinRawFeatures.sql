select 	m5.date_time as time,
		m5.close,
		m5.scaled_close,
		m5.wl,
		m5.zeit_index,
		m5.round_lot,
		m5.bb_h, 
		m5.bb_l,
		m5.macd,
		m5.macd_d,
		m5.rsi,
		m30.bb_h as bb_h_30min,
		m30.bb_l as bb_l_30min,
		m30.macd as macd_30min,
		m30.macd_d as macd_d_30min,
		m30.rsi as rsi_30min,
		h6.bb_h as bb_h_6h,
		h6.bb_l as bb_l_6h,
		h6.macd as macd_6h,
		h6.macd_d as macd_d_6h,
		h6.rsi as rsi_6h		
from min5_candle m5
left join min30_candle m30
on (m5.date_time = m30.date_time and m5.currency = m30.currency)
left join h6_candle h6
on (m5.date_time = h6.date_time and m5.currency = m30.currency)
where m5.currency = 'EURUSD'
order by m5.date_time desc