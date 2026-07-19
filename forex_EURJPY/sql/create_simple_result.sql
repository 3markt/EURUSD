DROP table if exists simple_result;
CREATE table if not exists simple_result (
	currency char(7) not null,
	timescale char(8) not null,
	tday char(10) not null,
	iteration int NOT NULL,
	epoch int NOT NULL,
	cnt_pos int,
	cnt_neg int,
	cnt_true_pos_pred int,
	cnt_true_neg_pred int,
	cnt_false_pos_pred int,
	cnt_false_neg_pred int,
	wl_pos float(32),
	wl_neg float(32),
	primary key (currency, timescale, tday, iteration, epoch)
);
	