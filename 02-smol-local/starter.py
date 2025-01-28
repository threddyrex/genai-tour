
#
# Example program that uses llama_index to run a query against a local model.
#
# embedding model: sentence-transformers/all-MiniLM-L6-v2 (https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
# llm: HuggingFaceTB/SmolLM2-135M (https://huggingface.co/HuggingFaceTB/SmolLM2-135M)
#
# code adapted from: https://docs.llamaindex.ai/en/stable/getting_started/starter_example_local/
# dataDir pulled from: https://github.com/bluesky-social/atproto-website/tree/main/src/app/%5Blocale%5D/specs
# prompt: "Please give me a three-sentence summary about DIDs."
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


#
# Load the docs. I'm using a subset of the ATProto docs.
#
Log("documents")
documents = SimpleDirectoryReader(dataDir, recursive=True).load_data()


#
# Configure embed_model
#
Log("Settings.embed_model")
# use all-MiniLM-L6-v2 from Hugging Face for embedding  model
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")


#
# Configure llm
#
Log("Settings.llm")
#Settings.llm = Ollama(model="llama3", request_timeout=360.0)
# declare llm, using HuggingFace HuggingFaceTB/SmolLM2-135M
Settings.llm = Ollama(model="smollm2:135m", request_timeout=360.0)


#
# Create the embeddings.
#
Log("VectorStoreIndex.from_documents()")
index = VectorStoreIndex.from_documents(documents)


#
# Run the query.
#
Log("index.as_query_engine()")
query_engine = index.as_query_engine()

Log("query_engine.query()")
response = query_engine.query(prompt)

Log(response)


