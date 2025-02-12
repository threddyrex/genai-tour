#
# Unit 1 of the hugging face agents course.
# from: https://huggingface.co/learn/agents-course/unit1/dummy-agent-library
#

import os
from huggingface_hub import InferenceClient
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


#
# Get client. The public version is used here.
#
#client = InferenceClient("meta-llama/Llama-3.2-3B-Instruct")
client = InferenceClient("https://jc26mwg228mkj8dw.us-east-1.aws.endpoints.huggingface.cloud")


#
# First attempt. No special tokens.
#
output = client.text_generation(
    "The capital of France is",
    max_new_tokens=100,
)

Log(output)


#
# Second attempt. Use special tokens.
#
prompt="""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
The capital of France is<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
output = client.text_generation(
    prompt,
    max_new_tokens=100,
)

Log(output)


#
# Third attempt. Use chat method.
#
output = client.chat.completions.create(
    messages=[
        {"role": "user", "content": "The capital of France is"},
    ],
    stream=False,
    max_tokens=1024,
)
Log(output.choices[0].message.content)



#
# Fourth attempt. Weather in London.
#

SYSTEM_PROMPT=r"""
Answer the following questions as best you can. You have access to the following tools:

get_weather: Get the current weather in a given location

The way you use the tools is by specifying a json blob.
Specifically, this json should have an `action` key (with the name of the tool to use) and an `action_input` key (with the input to the tool going here).

The only values that should be in the "action" field are:
get_weather: Get the current weather in a given location, args: {"location": {"type": "string"}}
example use : 

{{
  "action": "get_weather",
  "action_input": {"location": "New York"}
}}

ALWAYS use the following format:

Question: the input question you must answer
Thought: you should always think about one action to take. Only one action at a time in this format:
Action:

$JSON_BLOB (inside markdown cell)

Observation: the result of the action. This Observation is unique, complete, and the source of truth.
... (this Thought/Action/Observation can repeat N times, you should take several steps when needed. The $JSON_BLOB must be formatted as markdown and only use a SINGLE action at a time.)

You must always end your output with the following format:

Thought: I now know the final answer
Final Answer: the final answer to the original input question

Now begin! Reminder to ALWAYS use the exact characters `Final Answer:` when you provide a definitive answer.
"""

prompt=f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{SYSTEM_PROMPT}
<|eot_id|><|start_header_id|>user<|end_header_id|>
What's the weather in London ?
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""


output = client.text_generation(
    prompt,
    max_new_tokens=200,
)

print(output)