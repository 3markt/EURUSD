CREATE table if not exists trade_log (
	currency char(7) not null,
	date_time char(16) not null,
	position real not null,
	close real not null,
	profit real not null,
	trade_cnt integer not null,
	primary key (currency, date_time)
);
	