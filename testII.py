from datasets import load_dataset

dataset = load_dataset("newsgroup")
dataset = load_dataset("CShorten/ML-ArXiv-Papers")["train"]

abstracts = dataset["abstract"]
titles = dataset["title"]