# Take the full dataset's timesteps and closing prices
dataset_timesteps = df_sp_500_price_closing.index.to_numpy()
dataset_prices = df_sp_500_price_closing['Price'].to_numpy()

# Testing out the windowing function for multiple records
dataset_full_windows, dataset_full_labels = make_windows(dataset_prices, window_size=WINDOW_SIZE_WEEK, horizon=HORIZON_DAY)
print(dataset_full_windows.shape, dataset_full_labels.shape)

print(pd.DataFrame(dataset_prices))
print(pd.DataFrame(dataset_full_windows))
print(pd.DataFrame(dataset_full_labels))

# Splitting into train and test sets
train_windows, test_windows, train_labels, test_labels = make_train_test_splits(dataset_full_windows, dataset_full_labels)
print(train_windows.shape, test_windows.shape, train_labels.shape, test_labels.shape)

# Saving train and test data
save_train_test_data(train_windows, train_labels, test_windows, test_labels, 
                     features=[f"day_{day}" for day in range(WINDOW_SIZE_WEEK)])

# Obtain the train and test datasets
train_dataset, test_dataset = gen_train_test_datasets(X_train=train_windows, y_train=train_labels,
                                                      X_test=test_windows, y_test=test_labels)
print(f"Train dataset: {train_dataset}", f"Test dataset: {test_dataset}")