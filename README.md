# Official Implementation of 'Tailoring the Shapley Value for In-context Example Selection towards Data Wrangling'.

# Technical Report
 
This repository contains the technical report of our paper 'Tailoring the Shapley Value for In-context Example Selection towards Data Wrangling', see ``
 
## Requirements
 
- Python 3.x
- Pytorch
- Numpy
- Sklearn

 
## Usage

We recommend updating the `API_key` variable in `API_key.txt` with your own OpenAI key. For reproducibility purpose, we give an OpenAI key as default.

To evaluate the UEA datasets using the commands:

ACSV:

`python CSV.py [dataset_name] --task [task_name]`

MCSV:

`python CSV.py [dataset_name] --task [task_name]`

BCSV:

`python CSV.py [dataset_name] --task [task_name]`

Use -h or --help option for the detailed messages of the other options, such as the hyper-parameters.
 
The main methods are implemented based on `fm-data-tasks` and `BatchER`. Thanks for the contribution!
