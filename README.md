# Official Implementation of 'Tailoring the Shapley Value for In-context Example Selection towards Data Wrangling'.

# Paper and Technical Report
 
This repository contains our paper currently under review at ICDE 2025, see `CSV.pdf` for details. 

Additional information can be found in the technical report, see `CSV_techreport.pdf` for details. 

# Task and Datasets
 
We use 11 datasets for 4 Data Wrangling tasks：Entity Matching, Data Imputation, Error Detection, and Schema Mapping.

| Dataset | Size | # Attr. |
| --- | --- | --- |
| **Fodors-Zagats** | 946 | 6 |
| **iTunes-Amazon** | 540 | 8 |
| **Beer** | 450 | 4 |
| **DBLP-ACM** | 12363 | 4 |
| **DBLP-GoogleScholar** | 28707 | 4 |
| **Amazon-Google** | 11460 | 3 |
| **Walmart-Amazon** | 10242 | 5 |
| **Buy** | 651 | 4 |
| **Restaurant** | 864 | 5 |
| **Adult** | 11000 | 13 |
| **Hospital** | 1000 | 19 |
| **Synthea** | 29637 | 9 |

 
## Requirements
 
- Python 3.x
- Pytorch
- Numpy
- Sklearn

 
## Usage

We recommend updating the `API_key` variable in `API_key.txt` with your OpenAI api key. For reproducibility purpose, we provided an OpenAI api key as default.

To evaluate the DW datasets, use the following commands:

ACSV:

`python ACSV.py [dataset_name]`

MCSV:

`python MCSV.py [dataset_name]`

CondAcc:

`python CondAcc.py [dataset_name]`

Use -h or --help option for the detailed messages of the other options, such as the hyper-parameters.

## To be updated

- The BCSV algorithm
- BatchER on Data Imputation, Error Detection, and Schema Mapping Tasks
- Parallel sampling
- CSV on other LLMs such as Llama-3
- RAG support for CSV
 
The main methods are implemented based on `fm-data-tasks` and `BatchER`. Thanks for the contribution!
