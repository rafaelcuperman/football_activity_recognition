import pandas as pd
import os
import random as rd
import numpy as np
from tensorflow.keras.models import load_model
import json
import matplotlib.pyplot as plt
import time
from scipy import stats, signal
from itertools import groupby
import matplotlib
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from collections import Counter

def classify_recording_batches(model, recording, window_length, order_of_sensors, order_of_sensors_acc, num_activities, thr_low_activity = 8.5, step = 10, batch_size = 100):
	'''
	Classifies the recording using sliding windows in batches and the deep learning model specified.
	Parameters:
		model: .h5 keras model
		recording: dataframe with the recording. Sensors are in the columns and each row is a timestep
		window_length (int): timesteps used for the sliding window
		order_of_sensors (list): list of the order of the sensors needed by the model
		order_of_sensors_acc (list): list of the order of only accelerometer sensors
		num_activities (int): Number of activities to detect excluding "low activity"
		thr_low_activity (double): Threshold used for the detection of 'low activitiy' category. Set to 0, None or False to not include 'low activity' as category 
		step (int): step of the sliding window in timesteps
		batch_size (int): number of windows to classify at the same time (batch)
		'''
	
	# If low activity will be recognized, the norm is needed
	if thr_low_activity:
		norm = np.linalg.norm(recording[order_of_sensors_acc], ord = 2, axis=1)
	
	# Organize the recording so it has the dimensions and structure needed for the model
	recording = recording[order_of_sensors].values
	recording = np.transpose(recording)
	recording = np.expand_dims(recording, axis = 2)
	recording = np.expand_dims(recording, axis = 0)

	# Initialize global variables
	start_window = 0 #Initial time of the first sliding window
	start_windows = [start_window] #List where the initial times of the sliding windows will be saved
	end = recording.shape[2]-window_length #Last initial time of sliding window allowed
	windows_to_evaluate = int(end/step) + 1 #Number of sliding windows to evaluate. Just needed to print some status
	scores = [] #Where the final scores will be saved
	total_num_windows = 0 #Total number of windows to be evaluated

	# Initialize batch variables
	windows_batch = np.zeros((batch_size, recording.shape[1], window_length, 1)) #Windows from the current batch
	stds_batch = np.zeros((batch_size)) #Std.deviations of the windows of the current batch
	num_windows = 0 #Number of windows of the current batch
	
	# Predict the activity in the window. Returns, for each window, the softmax score for each activity
	start = time.time() # Start timer
	while start_window <= end:
		# Print status
		if total_num_windows % 1000 == 0:
			print('Processed {}/{} windows'.format(total_num_windows, windows_to_evaluate))
		
		# End of batch
		if num_windows == batch_size:
			scores_batch = -np.ones((batch_size,num_activities)) #Initialize scores of the windows of the batch as low activity
			num_windows = 0 #Restart the number of windows of the batch to zero
			
			if thr_low_activity:
				low_activity_batch = stds_batch < thr_low_activity #Check windows with standard deviation lower than the threshold
				windows_batch_activities = windows_batch[~low_activity_batch] #Windows with activities to classify (when low_activity_batch is False, meaning that it is not low activity)
				if windows_batch_activities.size > 0: #If there are non-low activities detected
					scores_batch[~low_activity_batch,:] = model.predict(windows_batch_activities, batch_size) #Predict and save in the respective indices of scores_batch
			else:
				scores_batch[:,:] = model.predict(windows_batch_activities, batch_size) #Predict all windows
				
			# Append to final scores
			scores.extend(scores_batch[:,:])
								
		# Get batch_size windows
		windows_batch[num_windows,:,:,:] = recording[:,:,start_window:start_window + window_length,:]
		stds_batch[num_windows] = np.std(norm[start_window:start_window + window_length])
		
		#Next window
		start_window += step
		start_windows.append(start_window)
		num_windows += 1
		total_num_windows += 1
	
	
	# Last batch in case it is smaller than batch_size
	if num_windows != batch_size:
		scores_batch = -np.ones((batch_size,num_activities)) #Initialize scores of the windows of the batch as low activity
		
		if thr_low_activity:
			low_activity_batch = stds_batch < thr_low_activity #Check windows with standard deviation lower than the threshold
			windows_batch_activities = windows_batch[~low_activity_batch] #Windows with activities to classify (when low_activity_batch is False, meaning that it is not low activity)
			if windows_batch_activities.size > 0: #If there are non-low activities detected
				scores_batch[~low_activity_batch,:] = model.predict(windows_batch_activities, num_windows) #Predict and save in the respective indices of scores_batch
		else:
			scores_batch[:,:] = model.predict(windows_batch_activities, num_windows) #Predict all windows
		
		# Append to final scores
		scores.extend(scores_batch[:num_windows,:])
		
	time_elapsed = time.time()-start #Time elapsed to classify the windows
	print('Processed {}/{} windows'.format(windows_to_evaluate, windows_to_evaluate))
	print('\nDone.\nTime elapsed to classify the recording: {} seconds\n'.format(time_elapsed))
	
	return scores, start_windows, time_elapsed, total_num_windows

def plot_predictions(recording, predictions, max_score, postprocessed_predictions, best_scores, predictions_no_outliers, labels_dict, thr_low_activity = 8.5, true_label_name = None, plot_colors = 'points'):
	'''
	Plots the predictions and postprocessed predictionds on top with some signals below
	Parameters:
		recording: recording. To plot it
		predictions: array or list of predictions for each timestep
		max_score: array with score of prediction
		postprocessed_predictions: array or list of postprocessed predictions
		best_scores: array with best scores after postprocessing
		predictions_no_outliers: array or list of predictions without outliers
		labels_dict: dictionary that gives the correspondence of label (in predictions) to activity
		thr_low_activity (double): Threshold used for the detection of 'low activitiy' category. Set to 0, None or False to not include 'low activity' as category 
		true_label_name: true activity of the recording. Set to None if unknown
		plot_colors: Select None to plot predictions without colors representing confidence of prediction, 'lines' to color lines or 'points' to color points.

	'''
	
	# Initialize plot
	fig, axes = plt.subplots(nrows=8, ncols=1, figsize=(14,20))
	
	# Add low activity to labels
	if thr_low_activity:
		labels_dict[-1] = 'low activity'
		
	# Plot predictions
	axes[0].plot(predictions)
	x_lim_fix_0 = axes[0].get_xlim()

	# Define colormap
	cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["yellow", "black"])
	norm = plt.Normalize(1/(max(labels_dict)+1), 1)

	if plot_colors == 'lines':  
	  axes[0].clear()

	  # Color the lines
	  points = np.array([range(len(predictions)), predictions]).T.reshape(-1, 1, 2)
	  segments = np.concatenate([points[:-1], points[1:]], axis=1)
	  lc = LineCollection(segments, cmap=cmap, norm=norm)
	  lc.set_array(np.array(max_score))
	  line = axes[0].add_collection(lc)

	elif plot_colors == 'points':
	  axes[0].clear()

	  #Color the points
	  axes[0].scatter(range(len(predictions)), predictions, c = max_score, cmap = cmap, norm = norm, s = 10)

	
	# Format plot
	labels_dict_predictions = {k:v for k,v in labels_dict.items() if v != "other high activity"} # Labels without "Other high activity". For the first plot
	axes[0].set_yticks(list(labels_dict_predictions.keys()))
	axes[0].set_yticklabels(labels_dict_predictions.values())
	axes[0].axes.get_xaxis().set_visible(False) 
	axes[0].set_title('Predictions')
	axes[0].grid(axis = 'y')
	
	
	# Plot postprocessed predictions
	axes[1].plot(postprocessed_predictions)
	x_lim_fix_1 = axes[1].get_xlim()
	y_lim_fix_1 = axes[1].get_ylim()
	
	if len(best_scores) > 0:
		axes[1].clear()
		
		# Color the lines
		points = np.array([range(len(postprocessed_predictions)), postprocessed_predictions]).T.reshape(-1, 1, 2)
		segments = np.concatenate([points[:-1], points[1:]], axis=1)
		lc = LineCollection(segments, cmap=cmap, norm=norm)
		lc.set_array(np.array(best_scores))
		line = axes[1].add_collection(lc)
		
	axes[1].set_yticks(list(labels_dict.keys()))
	axes[1].set_yticklabels(labels_dict.values())
	axes[1].axes.get_xaxis().set_visible(False)
	axes[1].set_title('Postprocessed predictions')
	axes[1].grid(axis = 'y')
	
	
	# Plot postprocessed predictions without outliers
	axes[2].plot(predictions_no_outliers)
	
	if len(best_scores) > 0:
		axes[2].clear()

		# Color the lines
		points = np.array([range(len(predictions_no_outliers)), predictions_no_outliers]).T.reshape(-1, 1, 2)
		segments = np.concatenate([points[:-1], points[1:]], axis=1)
		lc = LineCollection(segments, cmap=cmap, norm=norm)
		lc.set_array(np.array(best_scores))
		line = axes[2].add_collection(lc)
	
	axes[2].set_yticks(list(labels_dict.keys()))
	axes[2].set_yticklabels(labels_dict.values())
	axes[2].axes.get_xaxis().set_visible(False)
	axes[2].set_title('Predictions with outlier removal')
	axes[2].grid(axis = 'y')


	# Plot signals
	recording.plot.line(ax = axes[3], y = ['leftShankAccX', 'leftShankAccY', 'leftShankAccZ'], color = ['blue', 'red', 'orange']);
	recording.plot.line(ax = axes[4], y = ['leftThighAccX', 'leftThighAccY', 'leftThighAccZ'], color = ['blue', 'red', 'orange']);
	recording.plot.line(ax = axes[5], y = ['pelvisAccX', 'pelvisAccY', 'pelvisAccZ'], color = ['blue', 'red', 'orange']);
	recording.plot.line(ax = axes[6], y = ['rightShankAccX', 'rightShankAccY', 'rightShankAccZ'], color = ['blue', 'red', 'orange']);
	recording.plot.line(ax = axes[7], y = ['leftShankAccX', 'leftShankAccY', 'leftShankAccZ'], color = ['blue', 'red', 'orange']);

	
	# Format plot  
	for ax in axes[3:]:
		ax.legend(loc='upper right')
		ax.yaxis.tick_left()
		ax.axes.get_xaxis().set_visible(False)
	axes[-1].axes.get_xaxis().set_visible(True)
	
	# Format axis of first three plots
	axes[0].set_xlim(x_lim_fix_0)
	axes[0].set_ylim(y_lim_fix_1)
	
	axes[1].set_xlim(x_lim_fix_1)
	axes[1].set_ylim(y_lim_fix_1)
	
	axes[2].set_xlim(x_lim_fix_1)
	axes[2].set_ylim(y_lim_fix_1)
	
	# Add true label if known
	if true_label_name:
		fig.suptitle('True label: {}'.format(true_label_name)) 

	fig.tight_layout(rect = [0, 0, 1, 0.98])

def postprocess_predictions(len_recording, labels_dict, predictions, max_score, start_windows, window_length, method = 'best_score', thr_low_activity = 8.5, thr_other_high = 0.5, score_other_high = 0.95):
	'''
	Postprocesses predictions to timesteps of recording so that recording and predictions have the same length.
	Method: 'interpolate': The prediction of each window is assigned to the last timestep of the window. 
		For the timesteps between the end of window i and window i+1, all those timesteps are assigned to the 
		prediction of window i+1
	Method: 'mode': All the timesteps of the window are assigned to the prediction of the window. The final prediction for
		each timestep is the mode of those predictions for the timestep
	Method: 'best_score': All the timesteps of the window are assigned to the prediction of the window. The final prediction for
		each timestep is the prediction with highest predictionconfidence score for the timestep
	Parameters:
		len_recording (int): number of timesteps of the recording
		labels_dict(dict): dictionary that gives the correspondence of label (in predictions) to activity
		predictions (list): list of predictions obtained from classify_recording()
		max_score (list): list of scores of predictions obtained from classify_recording()
		start_windows (list): list of initial timesteps of windows obtained from classify_recording()
		window_length (int): length of sliding window in timesteps
		method (string): method of postprocessing
		thr_low_activity (double): threshold used for the detection of 'low activitiy' category. Set to 0, None or False to not include 'low activity' as category 
		thr_other_high (double): a prediction with score smaller than this value is considered "other high activity"
		score_other_high (double): once a prediction is considered as "other high activity", it will have score with value score_other_high. Set to None to not impose a score to these activities
	'''
		
	# Number of possible activities to detect
	number_labels = len(labels_dict) + 1 if thr_low_activity else len(labels_dict)
	
	# A prediction with score smaller than thr_other_high is considered "other high activity"
	if thr_other_high:
	  predictions = [max(labels_dict) if max_score[i] < thr_other_high else predictions[i] for i in range(len(predictions))]
	  if score_other_high:
	  	max_score = [score_other_high if max_score[i] < thr_other_high else max_score[i] for i in range(len(max_score))]
	
	# The prediction of each window is assigned to the last timestep of the window. 
	# For the timesteps between the end of window i and window i+1, all those timesteps are assigned to the 
	# prediction of window i+1
	if method == 'interpolate':
		preds = np.zeros(len_recording)
		for i, start_window in enumerate(start_windows):
			if i == 0:
				preds[0:window_length] = predictions[0]
			elif start_window == start_windows[-1]:
				preds[start_window:] = predictions[-1]
			else:
				preds[window_length + start_windows[i-1]:window_length + start_windows[i]] = predictions[i]
		best_score = []
		
	# All the timesteps of the window are assigned to the prediction of the window. The final prediction for
	# each timestep is the mode of those predictions for the timestep
	elif method == 'mode':
		preds = np.zeros((number_labels, len_recording))
		for i, start_window in enumerate(start_windows):
			if start_window == start_windows[-1]:
				preds[predictions[-1], start_window:] += 1
			else:
				preds[predictions[i],start_window:start_window + window_length] += 1

		preds = np.argmax(preds, axis = 0)
		preds = np.where(preds == len(labels_dict), -1, preds) # Replace low activity with -1
		
		best_score = []
	
	# All the timesteps of the window are assigned to the prediction of the window. The final prediction for
	# each timestep is the prediction with highest confidence score for the timestep
	elif method == 'best_score':
		best_score = np.zeros((number_labels, len_recording))
		for i, start_window in enumerate(start_windows):
			if start_window == start_windows[-1]:
				best_score[predictions[-1], start_window:] = [x if x>max_score[-1] else max_score[-1] for x in best_score[predictions[-1], start_window:]]
			else:
				best_score[predictions[i],start_window:start_window + window_length] = [x if x>max_score[i] else max_score[i] for x in best_score[predictions[i],start_window:start_window + window_length]]
		
		preds = np.argmax(best_score, axis = 0)
		preds = np.where(preds == len(labels_dict), -1, preds) # Replace low activity with -1
		
		best_score = np.max(best_score, axis = 0)
	
	else:
		print("Incorrect mode parameter: Choose 'mode', 'best_score' or 'interpolate'")
	
	return preds, best_score

def outliers_supression(predictions, thr_low_activity = 8.5, min_length_activity = 50):
	'''
	Remove outliers from predictions. 
	An outlier is a label that appears less than min_length_activity timesteps consecutively
	Parameters:
		predictions (list): List of predictions
		thr_low_activity (double): threshold used for the detection of 'low activitiy' category. Set to 0, None or False to not include 'low activity' as category 
		min_length_activity (int): minimum consecutive timesteps that a label must appear so it is not considered outlier
	'''
	
	# Grouped predictions returns the list of tuples of the number of consecutive predictions
	# For example, if predictions is [-1, -1, -1, 0, 0, -1, -1, -1, -1, 2, -1, -1]
	# The output of this line would be [(-1, 3), (0, 2), (-1, 5), (2, 1), (-1, 2)]
	grouped_predictions = [(k, sum(1 for i in g)) for k,g in groupby(predictions)]
	
	filtered_predictions = []
	for i, v in enumerate(grouped_predictions):
		#If there are less than min_length_activity consecutive predictions of a label, replace with the previous label
		if v[1] < min_length_activity: # The label appears less than min_length_activity consecutive times
			if i == 0: # If it is the first detected label of the recording
				if thr_low_activity: # If low activity is a valid category, set the correct label as low activity
					previous_ok = -1 # 
				else: # If low activity is not a valid category, assume that the first label is correct
					previous_ok = v[0]
		else: # The label appears at least min_length_activity consecutive times.
			previous_ok = v[0] # The label is correct
		filtered_predictions.extend([previous_ok]*v[1])

	predictions_no_outliers = filtered_predictions
	
	return predictions_no_outliers

def summary_of_activities(predictions, labels_dict, sampling_freq = 500):
	'''
	Prints the summary of activities of a recording.
	Example:
	Low activity: 3 seconds
	Run: 1.2 seconds
	Low activity: 2.5 seconds
	Total: 6.7 seconds
	Parameters:
		predictions (list of ints): list of lables of predictions
		labels_dict(dict): dictionary that gives the correspondence of label (in predictions) to activity
		sampling_freq (double): sampling frequency
	'''
	
	grouped_predictions = [(k, sum(1 for i in g)) for k,g in groupby(predictions)]
	
	print('Summary of activities for recording:')
	for label, duration in grouped_predictions:
		print('{}: {:.2f} seconds'.format(labels_dict[label], duration/sampling_freq))
	print('__________________________________________')
	print('Total: {:.2f} seconds'.format(len(predictions)/sampling_freq))
	
	print('\nNumber of times each activity was made:')
	count_labels = Counter(elem[0] for elem in grouped_predictions)
	for label, count in count_labels.items():
		if label != -1:
			print('{}: {}'.format(labels_dict[label], count))

	return grouped_predictions

def prepare_recording(recording, df, sampling_freq_original, order_of_sensors, file_labels_model, thr_low_activity = 8.5, sampling_freq = 500, normalize = False):
	'''
	Prepares recording for the sliding window evaluation
	Parameters:
		recording (pandas dataframe): recording to be prepared. Each column is a sensor
		df (pandas dataframe): full recording. Used for normalization
		sampling_freq_original (int): sampling frequency in Hz of the input recording
		order_of_sensors (list of strings): list with the required order of sensors
		file_labels_model (string): path of the file where the labels used by the model are
		thr_low_activity (double): Threshold used for the detection of 'low activitiy' category. Set to 0, None or False to not include 'low activity' as category 
		sampling_freq (int): required sampling frequency in Hz. Default 500
		normalize (boolean): Whether to normalize or not each sensor to have max absolute value 1. Default False
	Returns:
		recording (pandas dataframe): recording prepared for the sliding window evaluation
		thr_low_activity (double): threshold for low activity detection. Can be modified with respect to the input
		labels_dict (dictionary): dictionary of labels used by model
		num_activities (int): number of activities to detect
	'''

	recording = recording[order_of_sensors]

	# Normalize
	if normalize:
		aabs = pd.concat([df.max(), df.min().abs()], axis=1).max(axis=1)
		recording = recording.div(aabs)
		if thr_low_activity:
			thr_low_activity /= aabs[aabs.index.str.contains('Acc')].mean()

	print('Threshold for low activity: {}'.format(thr_low_activity))
		
	# Resampling of recording
	if sampling_freq_original != sampling_freq:
		print('Sampling frequency of signal is {}Hz but required sampling frequency is {}Hz. Resampling...\n'.format(sampling_freq_original, sampling_freq))
		recording = signal.resample(recording.values, int(len(recording)/sampling_freq_original*sampling_freq))
		print('Done resampling\n')
		
	recording = pd.DataFrame(recording, columns = order_of_sensors)
		
	print('Length of recording: {} seconds\n'.format(len(recording)/sampling_freq))
			 
	# Dictionary of labels used by model
	with open(file_labels_model) as json_file: 
		labels_dict = json.load(json_file) 
	labels_dict = {v:k for k,v in labels_dict.items()}

	num_activities = len(labels_dict.keys())

	return recording, thr_low_activity, labels_dict, num_activities

def recognition(model, recording, order_of_sensors, order_of_sensors_acc, num_activities, labels_dict, thr_low_activity, method = 'best_score', step = 50e-3, min_length_activity = 0.31, sampling_freq = 500, time_window = 1, score_low = 0.98, thr_other_high = 0.5, score_other_high = 0.95, batch_size = 128, true_label_name = None, plot_colors = 'points', plot = True, text = True):
	'''
	Performs the sliding window evaluation of a recording using a model
	Parameters:
		model (keras model): model used to do the evaluation
		recording (pandas dataframe): recording prepared for the sliding window evaluation
		order_of_sensors (list of strings): list with the required order of sensors
		order_of_sensors_acc (list of strings): list with only the accelerometer sensors
		num_activities (int): number of activities to detect
		labels_dict (dictionary): dictionary of labels used by model
		thr_low_activity (double): threshold used for the detection of 'low activitiy' category
		method (string): method of postprocessing. See postprocess_predictions function
		step (double): step in seconds of the sliding windows. If -1, the step is 1 timestep. Must be larger than 1/sampling_freq. Default 50e-3
		min_length_activity (double) minimum consecutive seconds that a label must appear so it is not considered outlier. Default 0.31
		sampling_freq (int): required sampling frequency in Hz. Default 500
		time_window (double): time of the windows in seconds. Default 1, 
		score_low (double): confidence score given to predictions considered as "low activity". Default 0.98
		thr_other_high (double): threshold to define "other high activity". Any prediction with score smaller than this, is considered "other high activity". Leave it as None to not include this category. Default 0.5
		score_other_high (double): confidence score given to predictions considered as "other high activity". Default 0.95. Set to None to not impose a score to these activities
		batch_size (int): number of windows to classify at the same time (batch)
		true_label_name (string): true activity of the recording. Set to None if unknown. Default None
		plot_colors (string): Select None to plot predictions without colors representing confidence of prediction, 'lines' to color lines or 'points' to color points. Default 'points'
		plot (boolean): whether to generate the plots of the predictions or not
		text (boolean): whether to generate the text summary of the predictions or not
	Returns:
		predictions_no_outliers (list): list of predictions
		best_scores (list): list scores of the predictions
		text_activities (list): list of tuples of predictions and their duration in timesteps
	'''

	# Window length
	window_length = sampling_freq * time_window
	print('Window length: {} seconds = {} timesteps'.format(time_window, window_length))

	# Step
	step = int(step*sampling_freq) if step != -1 else 1
	step = 1 if step == 0 else step
	print('Step size: {} seconds = {} timesteps\n'.format(step/sampling_freq, step))

	# Perform the classification with the model
	scores, start_windows, time_elapsed, num_windows = classify_recording_batches(model, recording, window_length, order_of_sensors, order_of_sensors_acc, num_activities, thr_low_activity = thr_low_activity, step = step, batch_size = batch_size)

	# Predictions by taking the argmax of the scores per window. Max score as the largest score of the predictions.
	predictions = []
	max_score = []
	for i in scores:
		if i[0] == -1:
			predictions.append(-1)
			max_score.append(score_low)
		else:
			predictions.append(i.argmax())
			max_score.append(i.max())
	max_score = np.array(max_score)

	# Add "Other high activity" if threshold is defined
	if thr_other_high:
		labels_dict[max(labels_dict) + 1] = 'other high activity'

	# Postprocess predictions
	start = time.time()
	postprocessed_predictions, best_scores = postprocess_predictions(len(recording), labels_dict, predictions, max_score, start_windows, window_length, method = method, thr_low_activity = thr_low_activity, thr_other_high = thr_other_high, score_other_high = score_other_high)

	# Remove outliers
	min_length_activity *= sampling_freq 
	predictions_no_outliers = outliers_supression(postprocessed_predictions, thr_low_activity = thr_low_activity, min_length_activity = min_length_activity)

	print('Postprocessing finished.\nTime elapsed to postprocess the predictions: {} seconds\n'.format(time.time()-start))

	# Plot postprocessed predictions
	if plot:
		plot_predictions(recording, predictions, max_score, postprocessed_predictions, best_scores, predictions_no_outliers, labels_dict, thr_low_activity, true_label_name = true_label_name, plot_colors = plot_colors)

	# Print summary of activities
	if text:
		text_activities = summary_of_activities(predictions_no_outliers, labels_dict, sampling_freq)
	else:
		text_activities = None

	return predictions_no_outliers, best_scores, text_activities

def main(model, recording, df, sampling_freq_original, order_of_sensors, order_of_sensors_acc, file_labels_model, thr_low_activity = 8.5, sampling_freq = 500, normalize = False, method = 'best_score', step = 50e-3, min_length_activity = 0.31, time_window = 1, score_low = 0.98, thr_other_high = 0.5, score_other_high = 0.95, batch_size = 128, true_label_name = None, plot_colors = 'points', plot = True, text = True):
	'''
	Parameters:
		model (keras model): model used to do the evaluation
		recording (pandas dataframe): recording to be prepared. Each column is a sensor
		df (pandas dataframe): full recording. Used for normalization
		sampling_freq_original (int): sampling frequency in Hz of the input recording
		order_of_sensors (list of strings): list with the required order of sensors
		order_of_sensors_acc (list of strings): list with only the accelerometer sensors
		file_labels_model (string): path of the file where the labels used by the model are
		thr_low_activity (double): Threshold used for the detection of 'low activitiy' category. Set to 0, None or False to not include 'low activity' as category 
		sampling_freq (int): required sampling frequency in Hz. Default 500
		normalize (boolean): Whether to normalize or not each sensor to have max absolute value 1. Default False
		method (string): method of postprocessing. See postprocess_predictions function
		step (double): step in seconds of the sliding windows. If -1, the step is 1 timestep. Must be larger than 1/sampling_freq. Default 50e-3
		min_length_activity (double) minimum consecutive seconds that a label must appear so it is not considered outlier. Default 0.31
		time_window (double): time of the windows in seconds. Default 1, 
		score_low (double): confidence score given to predictions considered as "low activity". Default 0.98
		thr_other_high (double): threshold to define "other high activity". Any prediction with score smaller than this, is considered "other high activity". Leave it as None to not include this category. Default 0.5
		score_other_high (double): confidence score given to predictions considered as "other high activity". Default 0.95. Set to None to not impose a score to these activities
		batch_size (int): number of windows to classify at the same time (batch)
		true_label_name (string): true activity of the recording. Set to None if unknown. Default None
		plot_colors (string): Select None to plot predictions without colors representing confidence of prediction, 'lines' to color lines or 'points' to color points. Default 'points'
		plot (boolean): whether to generate the plots of the predictions or not
		text (boolean): whether to generate the text summary of the predictions or not
	Returns:
		predictions_no_outliers (list): list of predictions
		best_scores (list): list scores of the predictions
		text_activities (list): list of tuples of predictions and their duration in timesteps
		labels_dict (dictionary): dictionary of labels used by model
	'''

	# Prepare recording
	recording, thr_low_activity, labels_dict, num_activities = prepare_recording(recording, df, sampling_freq_original, order_of_sensors, file_labels_model, thr_low_activity = thr_low_activity, sampling_freq = sampling_freq, normalize = normalize)

	# Evaluate recording
	predictions_no_outliers, best_scores, text_activities = recognition(model, recording, order_of_sensors, order_of_sensors_acc, num_activities, labels_dict, thr_low_activity, method = method, step = step, min_length_activity = min_length_activity, sampling_freq = sampling_freq, time_window = time_window, score_low = score_low, thr_other_high = thr_other_high, score_other_high = score_other_high, batch_size = batch_size, true_label_name = true_label_name, plot_colors = plot_colors, plot = plot, text = text)

	return predictions_no_outliers, best_scores, text_activities, labels_dict

if __name__ == '__main__':
	return None