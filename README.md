This is the repository for implementation details and technical reports of our paper 'Tailoring the Shapley Value for In-context Example Selection towards Data Wrangling'. 

Main Results of CSV on all datasets can be reproduced by running gpt-AC.py, including the intermediate result during sampling methods. Key parameters are given in the parameters.txt.

We recommend you to update with your own OpenAI account by changing the API_key variable in gpt-AC.py. You can try out CSV on any opensource LLM by running opensource-AC.py for free, we will update a version on our server when this link is no longer anonymous. 

Comparison Results 

Zero, Manual, and BatchER can be reproduced by running gpt-inference.py. The TaskSOTA results are from the original paper, you can go to their official link for reproducing results. 

Our implementation of AutoEM baseline and some other functions, like parrallel sampling with multithreading, will be updated soon.

The main methods are implemented based on fm-data-tasks and BatchER. Thanks for the contribution!
