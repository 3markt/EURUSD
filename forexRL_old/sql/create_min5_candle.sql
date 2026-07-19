DROP table if exists min5_candle;

CREATE table if not exists min5_candle (
	currency char(7) not null,
	date_time char(16) not null,
	close float(8) not null,
	scaled_close float(8),
	wl float(8) not null,
	zeit_index float(8) not null,
	round_lot float(8) not null,
	bb_h float(8),
	bb_l float(8),
	macd float(8),
	macd_d float(8),
	rsi float(8),
	primary key (currency, date_time)
);
	