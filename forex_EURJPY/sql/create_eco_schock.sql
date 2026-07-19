drop table if exists eco_schock;
CREATE table if not exists eco_schock (
	currency char(7) not null,
	date_time char(16) not null,
	name char(64) not null,
	feed_name char(64) not null,
	actual float(8) not null,
	forecast float(8) not null,
	primary key (currency, date_time, name, feed_name)
);