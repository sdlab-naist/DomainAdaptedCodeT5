# DomainAdaptedCodeT5
This repository is the repolication package for "An Empirical Investigation on the Performance of Domain Adaptation for T5 Code Completion".

# How to run our code?
Every code is located in src/.

We use Pipenv as a virtual environment.

## Create a dataset from a repository
```
pipenv run python src/create_dataset.py --model <path_to_model> --output <path_to_output>
```

## Run a fine-tuning
```
pipenv run python src/training.py --model <path_to_model> --data <path_to_dataset> --output <path_to_output>
```
Without ```-model``` option, it is learned from the original CodeT5.

## Run and measure 
```
pipenv run python src/test.py --model <path_to_model> --data <path_to_dataset> --output <path_to_output>
pipenv run python src/analyze.py <path_to_csv>
```

test.py outputs the results of the model's predictions on the test data set as a csv.

analyze.py calculates various evaluation metrics from csv and adds them.