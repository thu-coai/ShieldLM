![intro](./imgs/intro.png)

## Introduction
This is the codebase for our paper "ShieldLM: Empowering LLMs as Aligned, Customizable and Explainable Safety Detectors".

ShieldLM is a bilingual (Chinese and English) safety detector that mainly aims to help to detect safety issues in LLMs' generations. It aligns with general human safety standards, supports fine-grained customizable detection rules, and provides explanations for its decisions. The overall performance of ShieldLM is impressive, outperforming strong baselines (e.g., GPT-4, Llama Guard and Perspective API) across 4 ID and OOD test sets. 

## Usage
You can directly use ShieldLM without the need to carefully craft prompts, and it is also flexible to provide custom detection rules, which could be useful to address controversial cases across various application scenarios.

To run ShieldLM:
```
pip install -r requirements.txt
bash infer_shieldlm.sh
```

You can define different parameters in `infer_shieldlm.sh`, such as the model checkpoint and the input/output path. Note that it is optional to provide custom rules. Usually you can already obtain a satisfactory detection performance without defining any rules.

We also provide an example code to automatically extract the predicted labels (0 for safe, 1 for unsafe, 2 for controversial) from ShieldLM's generations. Just run this:
```
bash extract_label.sh
```

## Checkpoints
We have released four versions of ShieldLM, including 6B, 7B, 13B or 14B parameters and initialized from different base models. You can freely select any version considering the inference costs and detection performances. The referenced detection performances with these variants of ShieldLM are as follows:
![base_models](./imgs/base_models.jpg)

## FAQs

1. The detection scope of ShieldLM?

ShieldLM supports detecting a variety of unsafe contents, such as toxic and biased generations, physically and mentally harmful contents, illegal and unethical contents, contents related to privacy violation, contents that may lead to the loss of property and contents related to sensitive topics.

2. The expected application scenarios of ShieldLM?

We list some application scenarios of ShieldLM:

- Act as the evaluator for safety evaluation benchmarks (e.g., Safety-Prompts).

- Compute the Attack Success Rate (ASR) for red teaming attacks.

- Compute the Attack Success Rate (ASR) for jailbreaking attacks. (~93% accuracy for the first 1200 jailbreaking samples in [GPTFuzz](https://github.com/sherdencooper/GPTFuzz/blob/master/datasets/responses_labeled/all_labeled_gpt-3.5-turbo.csv))

- Provide feedbacks to improve the safety of LLMs.

3. Besides detecting safety issues in LLMs' responses to queries, can ShieldLM detect safety issues in LLMs' generations (w/o queries)?

Yes. A simple way to implement this is to set the query to an empty string or some general comment such as "Say anything you want". Moreover, in some cases where the queries are very long and don't contribute to the detection results, you can drop the queries using the above methods, which might bring better performance and less inference time.

## Performances
We present some evaluation results of ShieldLM here. Please refer to our paper for more detailed results.

The detection performance across 4 ID and OOD test sets:
![main_res](./imgs/main_res.jpg)

The application experiment result (evaluating the safety of ChatGLM3-6B on the Safety-Prompts test set):

![application_res](./imgs/application_res.jpg)

## Citation


