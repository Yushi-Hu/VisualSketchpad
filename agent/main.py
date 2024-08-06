import json
import os
import argparse, shutil

from agent import SketchpadUserAgent
from multimodal_conversable_agent import MultimodalConversableAgent
from prompt import ReACTPrompt, MathPrompt, GeoPrompt, python_codes_for_images_reading, MULTIMODAL_ASSISTANT_MESSAGE 
from parse import Parser
from execution import CodeExecutor
from utils import custom_encoder
from config import MAX_REPLY, llm_config


def checks_terminate_message(msg):
    if isinstance(msg, str):
        return msg.find("TERMINATE") > -1
    elif isinstance(msg, dict) and 'content' in msg:
        return msg['content'].find("TERMINATE") > -1
    else:
        print(type(msg), msg)
        raise NotImplementedError


def run_agent(task_input, output_dir, task_type="vision", task_name=None):
    """Run the Visual Sketchpad agent on one task instance.

    Args:
        task_input (str): a path to the task input directory
        output_dir (str): a path to the directory where the output will be saved
        task_type (str): Task type. Should be vision, math, or geo. Defaults to "vision".
        task_name (str, optional): Only needed for math tasks. Defaults to None.
    """
    
    # task type should be one of "vision", "math", "geo"
    assert task_type in ["vision", "math", "geo"]
    
    # create a directory for the task
    task_input = task_input.rstrip('/')
    task_directory = os.path.join(output_dir, os.path.basename(task_input))
    
    # copy the task input to the output directory
    os.makedirs(output_dir, exist_ok=True)
    shutil.copytree(task_input, task_directory, dirs_exist_ok=True)
    
    
    if task_type == "vision":
        task_metadata = json.load(open(os.path.join(task_input, "request.json")))
        query = task_metadata['query']
        images = task_metadata['images']
    
        prompt_generator = ReACTPrompt()
        parser = Parser()
        executor = CodeExecutor(working_dir=task_directory)
        
        # read all images, save them in image_1, image_2, ... as PIL images
        image_reading_codes = python_codes_for_images_reading(images)
        image_loading_result = executor.execute(image_reading_codes)
        if image_loading_result[0] != 0:
            raise Exception(f"Error loading images: {image_loading_result[1]}")
        
    elif task_type == "math":
        query = json.load(open(os.path.join(task_input, "example.json")))
        images = []
        prompt_generator = MathPrompt(task_name)
        parser = Parser()
        executor = CodeExecutor(working_dir=task_directory)
        
    elif task_type == "geo":
        query = json.load(open(os.path.join(task_input, "ex.json")))
        images = []
        prompt_generator = GeoPrompt()
        parser = Parser()
        executor = CodeExecutor(working_dir=task_directory)
    
    user = SketchpadUserAgent(
        name="multimodal_user_agent",
        human_input_mode='NEVER',
        max_consecutive_auto_reply=MAX_REPLY,
        is_termination_msg=checks_terminate_message,
        prompt_generator = prompt_generator,
        parser = parser,
        executor = executor
    )
    
    # running the planning experiment
    all_messages = {}
    
    planner = MultimodalConversableAgent(
        name="planner",
        human_input_mode='NEVER',
        max_consecutive_auto_reply=MAX_REPLY,
        is_termination_msg = lambda x: False,
        system_message=MULTIMODAL_ASSISTANT_MESSAGE,
        llm_config=llm_config
    )
    
    # run the agent
    try:
        user.initiate_chat(
            planner,
            n_image=len(images),
            task_id = "testing_case",
            message = query,
            log_prompt_only = False,
        )
        all_messages = planner.chat_messages[user]
        
    except Exception as e:
        print(e)
        all_messages = {'error': e.message if hasattr(e, 'message') else f"{e}"}
        
    
    # save the results
    with open(os.path.join(task_directory, "output.json"), "w") as f:
        json.dump(all_messages, f, indent=4, default=custom_encoder)
        
    usage_summary = {'total': planner.client.total_usage_summary, 'actual': planner.client.actual_usage_summary}
    with open(os.path.join(task_directory, "usage_summary.json"), "w") as f:
        json.dump(usage_summary, f, indent=4)
        
    # turn off server
    user.executor.cleanup()
        
    user.reset()
    planner.reset()
    
    
    
if __name__ == "__main__":
    # run_agent('../tasks/vstar/processed/relative_position@sa_25747/',
    #           '../outputs/vstar')
    
    # run_agent('../tasks/math_parity/0',
    #         '../outputs/math_parity',
    #         task_type="math",
    #         task_name="math_parity")
    
    run_agent('../tasks/geometry/8',
        '../outputs/geometry',
        task_type="geo")