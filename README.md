# Official Implementation and Technical Reports
 
This repository contains the official implementation and technical reports of 'Tailoring the Shapley Value for In-context Example Selection towards Data Wrangling'.
 
## Requirements
 
- Python 3.x
- Pytorch
- Numpy
- Sklearn

 
## Running
 
The main results of ACSV and MCSV on all datasets can be reproduced by running `gpt-AC.py`, including the intermediate results during sampling methods.
 
We recommend updating the `API_key` variable in `gpt-AC.py` with your own OpenAI account for optimal performance.
 
- Zero, Manual, SC(Sample Cluster) can be reproduced by running `gpt-inference.py` from `fm-data-tasks`.
- The TaskSOTA and BatchER results are from the original paper. You can visit their official link for reproducing those results.
 
Our implementation of the AutoEM baseline and some other functions, such as BatchCSV, parallel sampling, our BatchER variant, will be updated soon.
 
The main methods are implemented based on `fm-data-tasks` and `BatchER`. Thanks for their contributions!
