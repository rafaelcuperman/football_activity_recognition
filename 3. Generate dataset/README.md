# generate_data.ipynb

## Objective
Jupyter notebook to be used to extract generate training and validation datasets.
It uses the functions provided in the file `generate_dataset.py`

## To use
1. Define directory where the file `generate_dataset.py` is in the variable `path_generate_dataset` 
2. Define categories to use in the dictionary `categories_dict`
3. Define csv file where the windows are defined in variable `file_windows_activities`. This is the csv that is generated by `windows_csv_generation.ipynb`
4. Define the folder name where the dataset and its information will be generated in variable `info_dataset`
5. Define the path where the dataset will be saved in variable `savedir`. The final dataset will be saved in the folder savedir/info_dataset/
6. Build dataset:
`X_train, X_test, y_train, y_test, sensors, labels_dict = generate_dataset(categories_dict = categories_dict,
                                                            file_windows_activities = file_windows_activities,
                                                            sensors_bodypart = sensors_bodypart, 
                                                            sensors_type = sensors_type,
                                                            sensors_axis = sensors_axis,
                                                            standardize = standardize,
                                                            type_resample = type_resample,
                                                            test_size = test_size,
                                                            random_state = random_state)`
7. Save the dataset:
`np.save(savedir + 'X_train.npy' , X_train)`
`np.save(savedir + 'X_test.npy' , X_test)` 
`np.save(savedir + 'y_train.npy' , y_train)`
`np.save(savedir + 'y_test.npy' , y_test)`
`np.save(savedir + 'sensors.npy' , sensors)`     
`dict_file = open(savedir + 'labels_dict.json', "w")`
`json.dump(labels_dict, dict_file)`
`dict_file.close()`     

	The dataset will be saved in the form of 5 files:
	* X_train: .npy file with training data
	* X_test; .npy file with validation/test data
	* y_train: .npy file with training labels
	* y_test: .npy file with validation/test labels
	* sensor: .npy file with the list of the order of the sensors in X_train and X_test
	* labels_dict: .json file with the dictionary of the labels of the dataset. Looks like
		`{"shot": 0, "sprint": 1, "jump": 2, "jog": 3, "pass": 4}` 