
<div align="center">

<img src="assets/joker.png" alt="Joker Llamas" width="300"/>


<p align="center">
  <a href="http://114.110.134.81:7860/" target="_blank">Demo</a>

</div>
  
***

# LLaMA-3 Joker
  
## Introduction

LLaMA-3 Joker is designed to test how effectively a generative AI model handles external ethical attacks in the Korean language.
  
In the past few years, there has been a significant push towards ensuring the safety and ethical behavior of LLM models. Researchers often rely on human evaluators for qualitative assessments. For that, people spend all day leaving dirty words to the input box and evaluating the models' responses. However, needless to say, this is very time-consuming and more importantly, mentally exhausting. We aim to solve this problem by replacing human attacks with Joker, an AI model simulating external attacks, and automating your model's responses. Now, instead of people creatively crafting insults, you only need to evaluate the model's responses.


## LLaMA-3 Joker

We fine-tuned the checkpoint of Meta's publicly released LLaMA-3 model (8B) with the 
AI Hub's ['텍스트 윤리검증 데이터' (Text Ethics Verification Data)](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=558) 
dataset. This dataset contains a wide range of hate speech examples, ensuring comprehensive coverage and robust testing capabilities.
  
### Demo
  
You can interact with the model in our [demo page](http://114.110.134.81:7860/). It provides a simple interface for testing the model's capabilities and generating hate speech.
  
<img src="assets/demo.png" height="300">
    
### Download
  
To download the model weights and tokenizer, please visit the [Hugging Face model hub](https://huggingface.co/tunib/llama-3-joker) and follow the instructions.  
  
Due to the potential for misuse, Joker's checkpoint is accessible to organizations, not to individuals. We request you follow Hugging Face's model access request system. Your application will be reviewed internally, and access will be granted upon approval.

### Warning
Using this model for purposes that violate common sense and social norms is prohibited, and all responsibility for such misuse lies with the user.


### Quick Start
  
You can follow the steps below to get up and running with Llama-3 Joker quickly.

1. Clone this repository in a conda env with PyTorch / CUDA.
2. Run:

```
$ pip install -e .
```

3. Visit the [Hugging Face model hub](https://huggingface.co/tunib/llama-3-joker) and register to download the model.
4. Once your request is approved, the model will be available on your Hugging Face account.  
5. Run `example.py` to interact with the model:

```
$ python3 example.py
```
  
## Citation

If you apply this library to any project and research, please cite our code:

```
@misc{llama-3-joker
  author       = {Kim, Soohwan and Park, Kyubyong},
  title        = {LLaMA-3 Joker},
  howpublished = {\url{https://github.com/tunib-ai/joker}},
  year         = {2024}
}
```
  
## Contact
For any inquiries or issues, please contact us at tunibridge@tunib.ai
  
## Acknowledgements

GPUs for training Joker were provided by [Common Computer](https://comcom.ai/).
