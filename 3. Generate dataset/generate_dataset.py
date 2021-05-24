import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def get_key_where_value(value, dictionary):
    '''
    Returns the key where certain value is
    Arguments: 
        value: value to look
        dictionary: dictionary to look in
    '''
    for k,v in dictionary.items():
        if value in v:
            return k


def resample(df, type_resample = 'under', random_state = None):
    '''
    Resamples the dataframe to balance the data
    Arguments:
        df: dataframe to be resampled
    	type_resample (string): 'under' to undersample of majority labels, 'over' to oversample of minority labels, 'no' to not resample
        random_state (int): random seed
    Returns:
        df_resampled: resampled dataframe
    '''
    
    # Random undersample of majority labels
    if type_resample == 'under':
        min_samples = df['Label'].value_counts().min()
        df_resampled = pd.DataFrame()
        for l in set(df['Label'].values):
            df_l = df[df['Label'] == l]
            df_l = df_l.sample(min_samples, replace = False, random_state = random_state)
            df_resampled = pd.concat([df_resampled, df_l], axis=0)
    
    # Random oversample of minority labels
    elif type_resample == 'over':
        max_samples = df['Label'].value_counts().max()
        argmax_samples = df_windows['Label'].value_counts().index[df_windows['Label'].value_counts().argmax()] # Label with max_samples samples
        df_resampled = pd.DataFrame()
        for l in set(df['Label'].values):
            df_l = df[df['Label'] == l]
            if l != argmax_samples:
                df_l = df_l.sample(max_samples, replace = True, random_state = random_state)
            df_resampled = pd.concat([df_resampled, df_l], axis=0)
    elif type_resample == 'no':
    	df_resampled = df
    else:
        raise "Invalid value for type_resample"
        
    df_resampled = df_resampled.reset_index(drop = True)   
    
    # Plot distribution
    plt.figure()
    df_resampled['Label'].value_counts().plot(kind='bar', title='Count resampled (target)');
    
    return df_resampled


def build_dataset(df, sensors_bodypart = 'all', sensors_type = ['Acc'], sensors_axis = 'all', standardize = False):
    '''
    Builds dataset (X matrix and y labels):
    Arguments:
        df: Dataframe with with the following columns ('File': path of the csv where the window will be extracted, 
                                                       'Start': initial timestep of window,
                                                       'End': final timestep of window,
                                                       'Label': label of the window)
        sensors_bodypart: list of sensor bodyparts to use. 
                          If set to 'all' (default), the list is ['leftShank', 'rightShank', 'leftThigh', 'rightThigh', 'pelvis']
        sensors_type: list of sensor types to use. If set to 'all', the list is ['Acc', 'Gyro', 'Mag']
        sensors_axis: list of sensor axis to use. If set to 'all, the list is ['X', 'Y', 'Z']
        standardize (boolean): whether to standardize each signal or not
    Returns:
        X (numpy array of size (number of windows, number of sensors, window length)): Array with the signals
        y (numpy array of size (number of windows, number of labels)): Array with the labels one-hot-encoded
        sensors (list): List with the order of the sensors in X
    '''
    
    sensors_bodypart = ['leftShank', 'rightShank', 'leftThigh', 'rightThigh', 'pelvis'] if sensors_bodypart == 'all' else sensors_bodypart
    sensors_axis = ['X', 'Y', 'Z'] if sensors_axis == 'all' else sensors_axis
    sensors_type = ['Acc', 'Gyro', 'Mag'] if sensors_type == 'all' else sensors_type
    
    # Create the list of sensors
    sensors = ['{}{}{}'.format(i,j,k) for i in sensors_bodypart for j in sensors_type for k in sensors_axis]
    
    window_length = df.loc[0,'End']-df.loc[0,'Start']
    num_windows = len(df)
    
    X = np.zeros((num_windows, window_length, len(sensors)))
    y = np.zeros(num_windows)
    for index, row in enumerate(df.itertuples()):

        df_temp = pd.read_csv(row.File)
        start = row.Start
        end = row.End
        X[index,:,:] = df_temp.loc[start:end-1,sensors].to_numpy()
        
        if standardize:
            scaler = StandardScaler()
            X[index,:,:] = scaler.fit_transform(X[index,:,:])
        
        y[index] = row.Label
        
        if index % 100 == 0:
            print('{}/{} windows processed'.format(index, num_windows))
            
    X = np.transpose(X, (0, 2, 1))             
    y = pd.get_dummies(y).values 
    
    print('Dataset generated\n')
    print(X.shape)

    return X, y, sensors


def generate_dataset(categories_dict, file_windows_activities = r'../Data/windows/times_windows_activities_1_0.75.csv',
					sensors_bodypart = 'all', sensors_type = ['Acc'], sensors_axis = 'all', standardize = False, type_resample = 'under',
					test_size = 0.3, random_state = None):
    '''
    Main function to generate X_train, X_test, y_train and y_test. Also generates the list of sensors used for X
    Arguments: 
    	categories_dict (dictionary): Dictionary where each key is the name of a label and each value is the list of original labels (integers) that are the respective label
    	file_windows_activities (string): CSV file with the following columns ('File': path of the csv where the window will be extracted, 
            																   'Start': initial timestep of window,
                                                                               'End': final timestep of window,
                                                                               'Label': label of the window)
        sensors_bodypart: list of sensor bodyparts to use. If set to 'all' (default), the list is ['leftShank', 'rightShank', 'leftThigh', 'rightThigh', 'pelvis']
        sensors_type: list of sensor types to use. If set to 'all', the list is ['Acc', 'Gyro', 'Mag']
        sensors_axis: list of sensor axis to use. If set to 'all, the list is ['X', 'Y', 'Z']
        standardize (boolean): whether to standardize each signal or not
        type_resample (string): 'under' to undersample of majority labels, 'over' to oversample of minority labels, 'no' to not resample
        test_size (double. 0.3 default): Percentage test data
        random_state (int. None default): Random state for train-test split
    Returns:
    	X_train (numpy array of size (number of windows*(1-test_size), number of sensors, window length)): Array with training data
    	X_test (numpy array of size (number of windows*(test_size), number of sensors, window length)): Array with testing data
    	y_train (numpy array of size (number of windows*(1-test_size), number of labels)): Array with training labels one-hot-encoded
    	y_test (numpy array of size (number of windows*(test_size), number of labels)): Array with testing labels one-hot-encoded
    	sensors (list): List with the order of the sensors in X_train and X_test
    '''
    
    # Organize labels
    list_categories = [i for j in categories_dict.values() for i in j]
    labels_dict = {i:j for j,i in enumerate(categories_dict.keys())}
    new_labels_dict = {i: labels_dict[get_key_where_value(i, categories_dict)] for i in list_categories}
    print('Labels: {}\n'.format(labels_dict))
    
    
    # Read windows csv
    df_windows = pd.read_csv(file_windows_activities)
    
    # Filter csv to only take desired activities
    df_windows = df_windows[df_windows['Label'].isin(list_categories)].reset_index(drop = True)

    # Rename labels to 0...L
    df_windows['Label'] = df_windows['Label'].map(new_labels_dict)
    # Plot distribution of labels
    plt.figure()
    df_windows['Label'].value_counts().plot(kind='bar', title='Count (target)');
    
    
    # Resample to balance dataset
    df_resampled = resample(df_windows, type_resample = type_resample, random_state = random_state)
    
    
    # Build dataset
    X, y, sensors = build_dataset(df_resampled, 
    	sensors_bodypart = sensors_bodypart, 
    	sensors_type = sensors_type, 
    	sensors_axis = sensors_axis,
    	standardize = standardize)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle = True)

    print('Order of sensors: {}\n'.format(sensors))
    print('Size of train dataset: {}'.format(X_train.shape))
    print('Size of test dataset: {}'.format(X_test.shape))
    
    return X_train, X_test, y_train, y_test, sensors, labels_dict


# This is only executed when ran from script
if __name__ == '__main__':
    return None