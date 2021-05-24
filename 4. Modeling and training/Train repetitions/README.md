# train_repetitions.ipynb
**Must be run in [Google Colab](http://colab.research.google.com/)**
## Objective
Jupyter notebook to train several repetitions of the deep models.
Generates json file with training and testing accuracies and losses

## To use
Follow the instructions in the file 
Uses the dataset that is created with the file `generate_data.ipynb` 
Each model has its own function where its architecture, parameters and learning rate scheduler are defined. Each one of those functions returns the respective model and learning rate schedule

The main function is
`metrics = train_repetitions(X, y, model_function_names_CNN, model_function_names_RNN, num_trainings = 1, epochs = 200, normalize = normalize)`
It returns a dictionary called metrics, which has the losses and accuracies of the trainings.

