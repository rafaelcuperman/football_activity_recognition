# activity_detection.ipynb

## Objective
Jupyter notebook to be used to perform activity detection.
Isolates activities from pre and post low activity periods.

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
	
	(gen_label is a integer that codifies the activity present in the recording. Their codification is defined in the parameter `categories`)
			
	* savefile
	File where the recordings with extracted activities will be saved. It will be a csv file where each row references to an isolated activity. 
		In that file, each row will have the following columns:
		
			File: file where the recording is
			Initial time index of activity
			Final time index of activity (-1 if it is the last index)
3. Main function: 
    to detect activities from all the csv files in in_dir and generate the respective csv file savefile
    `extracted_activities, samples_activities, times_activities = extract_activities(in_dir, 
                                                                                    margin = margin, 
                                                                                    only_lengths = True, 
                                                                                    num_steps = num_steps, 
                                                                                    savefile = savefile)`
                       
## Additional functionalities (optional)
* Show an example of the procedure
	1. Choose a random recording from the in_dir folder
	2. Calculate norms:
	`norms, times, thr = calculate_norms(df, normalize = normalize)`
	3. Find initial and final times of activities:
	`initial, final = times_activity(norms, thr, num_steps = num_steps)`
	4. Plot initial and final times for each sensor:
	`plot_initial_final_timeseries(norms, times, thr, initial, final, plot_thr = True)`
	5. Show the final isolated activity:
	`show_examples(in_dir, normalize = normalize, filenames = [os.path.split(example)[-1]], margin = margin, num_steps = num_steps)`

* Show examples of detected activities
	1. Select number of examples to be plotted:
	`num_random_examples = 5`
	2. Plot examples: 
	`show_examples_from_file(in_dir, savefile, filenames = filenames, num_random_examples = num_random_examples)`