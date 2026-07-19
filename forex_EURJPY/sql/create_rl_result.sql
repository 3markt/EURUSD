DROP table if exists rl_result;
CREATE table if not exists simple_result (
	currency char(7) not null,
	tday char(10) not null,
	iteration int NOT NULL,
	profit_long float(32),
	profit_short float(32),
	cnt_true_pos_pred int,
	cnt_true_neg_pred int,
	cnt_false_pos_pred int,
	cnt_false_neg_pred int,
	wl_pos float(32),
	wl_neg float(32),
	primary key (currency, timescale, tday, iteration, epoch)
);
	