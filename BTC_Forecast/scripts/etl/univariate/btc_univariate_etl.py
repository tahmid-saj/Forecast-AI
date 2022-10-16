# Take the full dataset's timesteps and closing prices
dataset_timesteps = df_btc_price_closing.index.to_numpy()
dataset_prices = df_btc_price_closing['Price'].to_numpy()

# Testing out the windowing function for multiple records
dataset_full_windows, dataset_full_labels = make_windows(dataset_prices, window_size=WINDOW_SIZE_WEEK, horizon=HORIZON_DAY)
print(dataset_full_windows.shape, dataset_full_labels.shape)

print(pd.DataFrame(dataset_prices))
print(pd.DataFrame(dataset_full_windows))
print(pd.DataFrame(dataset_full_labels))

# Testing out the train/test split function
train_windows, test_windows, train_labels, test_labels = make_train_test_splits(dataset_full_windows, dataset_full_labels)
print(train_windows.shape, test_windows.shape, train_labels.shape, test_labels.shape)