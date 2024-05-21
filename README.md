This is the implementation of our paper 'CSV: Tailoring the Shapley Value for In-context Example Selection and Beyond'. Main Results of CSV on all datasets can be reproduced by running gpt-AC.py, including the intermediate result during sampling methods. Key parameters are given in the parameters.txt.

We recommend you to update with your own OpenAI account by changing the API_key variable in gpt-AC.py. We recommend at least 5 usd in your account for most datasets. You can try out CSV on any opensource LLM by running opensource-AC.py for free, we will update a version on our server when this link is no longer anonymous. 

Comparison Results of Zero, Manual, and can be reproduced by running gpt-inference.py. The Ditto and BatchER results are from the original paper, you can go to their official link for reproducing results. 

Our implementation of AutoEM baseline and some other functions, like parrallel sampling with multithreading, will be updated soon.

The main methods are implemented with fm-data-tasks and BatchER. Thanks for the contribution!
