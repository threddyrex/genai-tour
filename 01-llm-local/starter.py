
#
# Example program that uses llama_index to run a query against a local LLM.
# I used a subset of the ATProto docs as the dataDir.
#
# code adapted from: https://docs.llamaindex.ai/en/stable/getting_started/starter_example_local/
# dataDir pulled from: https://github.com/bluesky-social/atproto-website/tree/main/src/app/%5Blocale%5D/specs
# prompt: "Can you tell me what you know about DIDs"
#

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

import logging
import sys
from datetime import datetime


def Log(msg):
    print(datetime.now().strftime("%H:%M:%S"), msg)

# Args
dataDir = sys.argv[1]
prompt = sys.argv[2]
verboseLogging = True if sys.argv[3] == "True" else False

Log("dataDir: " + dataDir)
Log("prompt: "+ prompt)
Log("verboseLogging: " + str(verboseLogging))

# Verbose logging
if(verboseLogging):
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))



# Load the docs. I'm using a subset of the ATProto docs.
Log("load_data()")
documents = SimpleDirectoryReader(dataDir, recursive=True).load_data()

Log("HuggingFaceEmbedding()")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

Log("Ollama()")
Settings.llm = Ollama(model="llama3", request_timeout=360.0)


# Create the embeddings.
Log("VectorStoreIndex.from_documents()")
index = VectorStoreIndex.from_documents(documents)

Log("index.as_query_engine()")
query_engine = index.as_query_engine()

# Run the query.
Log("query_engine.query()")
response = query_engine.query(prompt)

Log(response)


