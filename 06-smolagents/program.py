
#
# intro to smolagents
# from: https://huggingface.co/blog/smolagents
#

from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel
import os
from datetime import datetime


def Log(msg):
    print("----------------------------------------")
    print(datetime.now().strftime("%H:%M:%S"), msg)
    print("----------------------------------------")


#
# get hftoken and show first/last char
#
os.environ["HF_TOKEN"] = os.sys.argv[1]
Log(os.environ["HF_TOKEN"][0] + "..." + str(len(os.environ["HF_TOKEN"])-2) + "..." + os.environ["HF_TOKEN"][len(os.environ["HF_TOKEN"]) - 1])


agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=HfApiModel())

agent.run("How many seconds would it take for a leopard at full speed to run through Pont des Arts?")
