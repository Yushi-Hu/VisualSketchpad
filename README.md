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

After the server is launched, please edit the gradio servers link in  [`agent/config.py`](https://github.com/Yushi-Hu/VisualSketchpad/blob/main/agent/config.py). Change the server addresses to yours.
```python
SOM_ADDRESS = "[YOUR SOM SERVER ADDRESS]"
GROUNDING_DINO_ADDRESS = "[YOUR GroundingDINO SERVER ADDRESS]"
DEPTH_ANYTHING_ADDRESS = "[YOUR Depth-Anything SERVER ADDRESS]"
```


## Quick Start

### Data
We preprocessed each task and put them into `tasks`. Each instance in each task has a separate folder. Some tasks are too big, so we put it in this [Google Drive Link](https://drive.google.com/file/d/1qtbfI7Q9B7pq-WR20q0-OE6OetJqoitS/view?usp=sharing). Please download, unzip, and put the content in the `tasks` folder.

### Run the agent
See `agent/quick_start_math.py` for a simple example of running the math tasks. As seen, the code is modularized. The key function is `run_agent` in `agent/main.py`, which use the agent to finish a task.
```python
from main import run_agent

# run a example for graph max flow. save the execution trace, answer, and usage summary under outputs/graph_maxflow
run_agent("../tasks/graph_maxflow/5", "../outputs/graph_max_flow", task_type="math", task_name="graph_maxflow")

# run a example for geometry. save the execution trace, answer, and usage summary under outputs/geometry
run_agent("../tasks/geometry/2079", "../outputs/geometry", task_type="geo")
```

After installing and setting up all the gradio servers, you can also try run the vision task agent in `agent/quick_start_vision.py`. The structure is similar:
```python
from main import run_agent

# run a example for vision tasks. save the execution trace to outputs/blink_spatial
run_agent("../tasks/blink_spatial/processed/val_Spatial_Relation_1", "../outputs/blink_spatial", task_type="vision")
```

We put the expected running outputs in `outputs` as reference.
