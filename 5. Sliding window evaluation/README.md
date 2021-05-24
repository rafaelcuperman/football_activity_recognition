# sliding_window_evaluation_colab.ipynb
**Must be run in [Google Colab](http://colab.research.google.com/)**
## Objective
Jupyter notebook to evaluate a recording with the pipeline

## To use
Follow the instructions in the file 
Uses the functions of the file `sliding_window_evaluation.py`. This file must be uploaded to Google Drive in advance.

The main function is
`predictions_no_outliers, best_scores, text_activities, labels_dict = swe_main(model = model, recording = recording, df = df, sampling_freq_original = sampling_freq_original, order_of_sensors = order_of_sensors, order_of_sensors_acc = order_of_sensors_acc, file_labels_model = file_labels_model, thr_low_activity = thr_low_activity, sampling_freq = sampling_freq, normalize = normalize, method = method, step = step, min_length_activity = min_length_activity, time_window = time_window, score_low = score_low, thr_other_high = thr_other_high, score_other_high = score_other_high, batch_size = batch_size, true_label_name = true_label_name, plot_colors = plot_colors, plot = plot, text = text)`

It returns the following:
- predictions_no_outliers: final predictions. Can be translated to actual activities using the dictionary `labels_dict`.
- best_scores: list of scores of the predictions
- text_activities: list of tuples of predictions and their duration in timesteps
- 	labels_dict: dictionary of labels used by model

The version `sliding_window_evaluation_local.ipynb` is exactly the same but can be used to run the model locally instead.