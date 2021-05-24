
# windows_csv_generation.ipynb

## Objective
Jupyter notebook to be used to extract windows for training and validation

## To use
1. Define parameters, especially the directories/files:
	* in_dir
	    folder where the recordings are. It has many csv files, each one with a recording.
		The csv files are separated by commas and each columns has the measurements of a signal.
		The names of the columns must be:
		
			time, 
			gen_label,
			leftShankAccX,
			leftShankAccY,
			leftShankAccZ,
			rightShankAccX,
			rightShankAccY,	
			rightShankAccZ,
			leftThighAccX,
			leftThighAccY,
			leftThighAccZ,
			rightThighAccX,
			rightThighAccY,
			rightThighAccZ,
			pelvisAccX,
			pelvisAccY,
			pelvisAccZ, 
			leftShankGyroX,
			leftShankGyroY,
			leftShankGyroZ,
			rightShankGyroX,
			rightShankGyroY,	
			rightShankGyroZ,
			leftThighGyroX,
			leftThighGyroY,
			leftThighGyroZ,
			rightThighGyroX,
			rightThighGyroY,
			rightThighGyroZ,
			pelvisGyroX,
			pelvisGyroY,
			pelvisGyroZ
			
		(gen_label is a integer that codifies the activity present in the recording)
	* times_file
	File where the recordings with extracted activities are saved. It is be a csv file where each row references to an isolated activity. 
		In that file, each row has the following columns:
		
			File: file where the recording is
			Initial time index of activity
			Final time index of activity (-1 if it is the last index)
			
	* times_low_activity
	File where the recordings with extracted activities are saved. It is be a csv file where each row references to an isolated activity
	In that file, each row has the following columns:
	
			File: file where the recording is
			Initial time index of low
			Final time index of low (-1 if it is the last index)

	* savefile
	File where the refrences to the extracted windows will be saved. It is a csv file where each row references to a window.
	In that file, each row will have the following columns:
		
		File: file where the window should be extracted from
		Start: initial index of window
		End: final index of window (-1 if it is the last index)
		Label: integer that codifies for the activity of the window (-1 if low activity). Comes from gen_label of in_dir files
	
3. Main function: 
    To generate the csv file savefile with the references to the extracted windows
    `windows_all(in_dir = in_dir, times_activities=times_file, times_low_activity = times_low_activity, savefile = savefile,
              window_length_seconds = window_length_seconds, sampling_freq = sampling_freq, overlap = overlap)`
                       
## Additional functionalities (optional)
* Run process only for high activities
	1. Define csv file where windows will be saved: `times_windows_activities`
	2. Generate and save windows:
	`windows_activities(in_dir = in_dir, times_file = times_file, savefile = times_windows_activities, 
                   window_length_seconds = window_length_seconds, sampling_freq = sampling_freq, overlap = overlap)`
* Run process only for low activities
	1. Define csv file where windows will be saved: `times_windows_low`
	2. Generate and save windows:
	`windows_low(in_dir = in_dir, times_low_activity = times_low_activity, savefile = times_windows_low,
              window_length_seconds = window_length_seconds, sampling_freq = sampling_freq, overlap = overlap)`
