# one_model.ipynb
**Must be run in [Google Colab](http://colab.research.google.com/)**
## Objective
Jupyter notebook to train a deep model and save it
Generates .h5 file with the model saved

## To use
Follow the instructions in the file 
Uses the dataset that is created with the file `generate_data.ipynb` 
Each model has its own function where its architecture, parameters and learning rate scheduler are defined. Each one of those functions returns the respective model and learning rate schedule

The main function is
`model = trainModel(X, y, selected_model, model_function_names_CNN, model_function_names_RNN, epochs = 200, normalize = normalize)`
It returns the trained model which then can be saved with the command `model.save('model.h5')`
`selected_model` is a string saying whith model will be trained. Choose a model from the keys of dictionary `model_function_names_CNN` or `model_function_names_RNN` 