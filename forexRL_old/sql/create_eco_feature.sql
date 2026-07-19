drop table if exists eco_feature;
CREATE table if not exists eco_feature (
	currency char(7) not null,
	date_time char(16) not null,
	name char(64) not null,
	actual float(8) not null,
	forecast float(8) not null,
	primary key (currency, date_time, name)
);