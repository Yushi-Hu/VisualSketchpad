# Visual Sketchpad  <img src="assets/icon.png" width="50" />
This repo contains evaluation code for the paper "[Visual Sketchpad: Sketching as a Visual Chain of Thought for Multimodal Language Models](https://arxiv.org/abs/2406.09403)"

[**🌐 Homepage**](https://visualsketchpad.github.io/) | [**📖 arXiv**](https://arxiv.org/abs/2404.12390) | [**📑 Paper**](https://arxiv.org/pdf/2406.09403.pdf) 

## 🔔News

 **🔥[2024-08-03]: Releasing the codes for Visual Sketchpad**

 ## Introduction

 ![Alt text](assets/teaser.jpg)


## Installation

Install the agent environment as follows:
```bash
conda create -n sketchpad python=3.9

pip install pyautogen==0.2.26
pip install 'pyautogen[jupyter-executor]'
pip install Pillow joblib matplotlib opencv-python numpy gradio gradio_client networkx scipy datasets
```
Set up your OpenAI API key in [`agent/config.py`](https://github.com/Yushi-Hu/VisualSketchpad/blob/main/agent/config.py). Edit the following:
```python
# set up the LLM for the agent
os.environ['OPENAI_API_KEY'] = '[YOUR OPENAI API KEY]'
os.environ["AUTOGEN_USE_DOCKER"] = "False"
llm_config={"cache_seed": None, "config_list": [{"model": "gpt-4o", "temperature": 0.0, "api_key": os.environ.get("OPENAI_API_KEY")}]}
```
Above is all it needs for math and geometry tasks. 

### Installing vision experts for computer vision tasks

For computer vision tasks, you also need to install the vision experts.
In this code base, each vision expert is a gradio server. You can set them up in other servers, and access them through web link. This allows you to run sketchpad agents on your computer, while all vision models running on another GPU server.
Follow [`vision_experts/installation.md`](https://github.com/Yushi-Hu/VisualSketchpad/blob/main/vision_experts/installation.md) to install and launch all the vision experts.
