import os

# set up the agent
MAX_REPLY = 10

# set up the LLM for the agent
os.environ['OPENAI_API_KEY'] = '[YOUR OPENAI API KEY]'
os.environ["AUTOGEN_USE_DOCKER"] = "False"
llm_config={"cache_seed": None, "config_list": [{"model": "gpt-4o", "temperature": 0.0, "api_key": os.environ.get("OPENAI_API_KEY")}]}


# use this after building your own server. You can also set up the server in other machines and paste them here.
SOM_ADDRESS = "http://localhost:8080/"
GROUNDING_DINO_ADDRESS = "http://localhost:8081/"
DEPTH_ANYTHING_ADDRESS = "http://localhost:8082/"
