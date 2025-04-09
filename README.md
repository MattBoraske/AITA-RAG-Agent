# r/AITA Conflict Evaluation Using a RAG Agent 

This repository is supplemental to the paper ['Context is Key: Aligning Large Language Models with Human
Moral Judgments through Retrieval-Augmented Generation'](https://github.com/MattBoraske/AITA-Judge/blob/main/paper/FLAIRS_38_Paper.pdf), presented at [FLAIRS-38](https://www.flairs-38.info/home) and published in [Florida Online Journals](https://journals.flvc.org/FLAIRS/).

## Overview
The project introduces an AI agent that evaluates interpersonal conflicts by using Retrieval-Augmented Generation (RAG) to:
1. Collect similar conflicts from a dataset
2. Use these conflicts as context to refine the LLM's judgment
3. Provide adaptable moral evaluations without costly fine-tuning

## Dataset
A [dataset](https://huggingface.co/datasets/MattBoraske/reddit-AITA-submissions-and-comments-multiclass) containing the top 50,000 submissions to the r/AmITheAsshole (r/AITA) subreddit from 2018-2022 was created, including the top ten comments for each post.

## Results
Using OpenAI's GPT-4o as the base LLM, two agents were developed:
1. Base: doesn't use RAG to refine its responses.
2. RAG: uses RAG to retrieve AITA conflicts to use as evidence to iteratively refine its response.

The RAG agent, demonstrated clear improvements over the Base Agent, as its accuracy increased from 77% to 84% and its Matthews correlation coefficient (MCC) improved from 0.357 to 0.469. Additionally, the generation of any toxic responses was practically eliminated. 

These findings demonstrate that integrating LLMs into RAG frameworks effectively improves alignment with human moral judgments while mitigating harmful language.

⚠️ __Note:__ As FLAIRS-38 is upcoming, the paper is temporarily included in this repository and will be replaced with a link to the official publication after the conference.
