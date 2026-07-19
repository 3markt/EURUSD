CREATE table if not exists forex_raw (
	currency char(7) not null,
	date_time char(16) not null,
	open float(8) not null,
	low float(8) not null,
	high float(8) not null,
	close float(8) not null,
	cnt_diff_vals integer not null,
	primary key (currency, date_time)
);
	