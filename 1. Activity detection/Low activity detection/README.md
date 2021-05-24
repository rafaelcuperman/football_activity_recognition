
# low_activity_detection.ipynb

## Objective
Jupyter notebook to be used to perform activity detection.
Saves periods of low activity that happen before and/or after isolated activities

## To use
1. Define parameters, especially the directories/files:
	* in_dir
	    folder where the recordings are. It has many csv files, each one with a recording.
		The csv files are separated by commas and each columns has the measurements of a signal.
		The names of the columns must be:
		
			time, 
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
			
	* times_file
	File where the recordings with extracted activities are saved. It is be a csv file where each row references to an isolated activity. 
		In that file, each row has the following columns:
		
			File: file where the recording is
			Initial time index of activity
			Final time index of activity (-1 if it is the last index)
			
	* times_low_activity
	File where the recordings with extracted activities will be saved. It will be a csv file where each row references to an isolated activity.
	In that file, each row will have the following columns:
	
			File: file where the recording is
			Initial time index of low
			Final time index of low (-1 if it is the last index)

3. Main function: 
    to detect low activities from all the csv files in in_dir and generate the respective csv file times_low_activities
    `save_low_activity(in_dir, times_file, times_low_activity)`
                       
## Additional functionalities (optional)
* Show examples of detected low activities
	`plot_examples(in_dir, times_low_activity, num_random_examples = 5)`