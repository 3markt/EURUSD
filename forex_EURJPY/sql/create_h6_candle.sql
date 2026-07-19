DROP table if exists h6_candle;
CREATE table if not exists h6_candle (
	currency char(7) not null,
	date_time char(16) not null,
	wl float(8) not null,
	bb_h float(8),
	bb_l float(8),
	macd float(8),
	macd_d float(8),
	rsi float(8),
	primary key (currency, date_time)
);
	