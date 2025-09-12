# Hallucination Detection on a Budget: Efficient Bayesian Estimation of Semantic Entropy

Link to our paper: 
> Kamil Ciosek, Nicolò Felicioni, Sina Ghiassian.
> [Hallucination Detection on a Budget: Efficient Bayesian Estimation of Semantic Entropy](https://arxiv.org/abs/2504.03579).

This repository contains the code required to completely reproduce the results presented in our paper. It is also intended to serve as a useful foundation for researchers who wish to build upon our work or use our data files for further study.

Notably, all our results can be reproduced on a standard personal laptop—no specialized hardware or infrastructure is required.

With this codebase, you can:
- Reproduce the results from our paper exactly.
- Run all experiments on a typical laptop.
- Leverage our preprocessed data (pickled files) to conduct independent research on semantic entropy estimation.

Our work extends the ideas introduced in the paper [Detecting Hallucinations in Large Language Models Using Semantic Entropy](https://www.nature.com/articles/s41586-024-07421-0). Accordingly, we used their [code](https://github.com/jlko/semantic_uncertainty) as a starting point. 

### Software Dependencies

This project was developed and tested with Python 3.11.2.

The following Python packages are required to run the code:
	•	numpy
	•	scikit-learn
	•	matplotlib
	•	scipy
	•	joblib

Additionally, the code uses standard Python libraries:
	•	os
	•	shutil
	•	pickle
	•	random
	•	functools
	•	collections

These dependencies can be installed via pip or managed through a virtual environment of your choice. Our code relies on Python 3.11 with PyTorch 2.5.1 and our systems runs Debian GNU/Linux 12 (Bookworm) operating system.

### Reproducing results from the paper (based on precomputed pickle files with LLM responses)

To reproduce all the plots and tables presented in our paper, you only need to run the following notebook:
```
/home/usr_name/dev/llm-hallucinatio/blob/master/notebooks/estimation_final.ipynb
```
The notebook has been designed to be lightweight and efficient, making it accessible to researchers without access to specialized hardware. Because it relies on precomputed responses from LLMs, it avoids the need for GPU acceleration. As a result, the full execution is expected to complete within 15 minutes or less on a typical personal laptop, enabling easy and reproducible analysis of our results.

## Generating Pickle Files
If you want to regenrate the pickle files (LLM responses) too, you can follow the following instructions. Note that this typically requires a GPU and takes several days.

To generate the required `.pkl` files, follow the steps below from the **root directory** of the repository:

### 1. Generate Answers

Run the following command:
```
python semantic_uncertainty/generate_answers.py
--model_name=Llama-3.2-3B-Instruct
--dataset=trivia_qa
--num_generations=100
--num_samples=1000
--no-compute_p_ik
```


* Replace `Llama-3.2-3B-Instruct` with your desired model name.
* Replace `trivia_qa` with your desired dataset name.

This will generate a log file in the root directory, with a name like:
```
log-2025-03-13 17:42:16.919802.txt
```


### 2. Extract the Run Hash

Open the log file you just created and look for a line near the top that looks like this:
```
2025-03-13 17:42:17 INFO creating RunStatusReporter for 86e1574943c649bb91dd3fe5
```


Here, `86e1574943c649bb91dd3fe5` is the **run hash** you'll use in the next step.

### 3. Update the Notebook Paths

Open one of the `new_extract_dataset_*.ipynb` notebooks inside the `notebooks/` directory.

Update the following file paths with your extracted run hash:

```
with open(f'/home/usr_name/dev/llm-hallucination/.aim/86e1574943c649bb91dd3fe5/uncertainty_measures.pkl', 'rb') as file:
```
and
```
with open(f'/home/usr_name/dev/llm-hallucination/.aim/86e1574943c649bb91dd3fe5/validation_generations.pkl', 'rb') as file:
```
### 4. Run the Notebook

Execute all cells in the notebook. This will generate the final `.pkl` file used for downstream analysis and figure generation, as reported in the paper.


## Authors

- [Kamil Ciosek](mailto:kamilc@spotify.com)
- [Nicolò Felicioni](mailto:nicolof@spotify.com)
- [Sina Ghiassian](mailto:sinag@spotify.com)
