# Including block reward to the full BTC closing price dataset
df_btc_price_block_reward = include_block_reward(df_btc_price=df_btc_price_closing)

# Plotting the BTC price and block reward over time
plot_block_reward_price_data(df_btc_price_block_reward=df_btc_price_block_reward)

# Obtain the full windows and labels with block reward included
dataset_full_windows_br, dataset_full_labels_br = make_windows_labels_multivariate(df_btc_price_block_reward)

# Obtain the train/test sets of the full windows and labels with block reward included
train_windows, test_windows, train_labels, test_labels = make_train_test_splits(windows=dataset_full_windows_br, labels=dataset_full_labels_br)

# Obtain the train and test datasets
train_dataset, test_dataset = gen_train_test_datasets(X_train=train_windows, y_train=train_labels,
                                                      X_test=test_windows, y_test=test_labels)
print(f"Train dataset: {train_dataset}", f"Test dataset: {test_dataset}")