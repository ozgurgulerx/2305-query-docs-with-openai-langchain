import openai
from openai.embeddings_utils import get_embedding, cosine_similarity

doc_reader = PdfReader('./Seneca-OntheShortnessofLife.pdf')

# read data from the file and put them into a variable called raw_text
raw_text = ''
for i, page in enumerate(doc_reader.pages):
    text = page.extract_text()
    if text:
        raw_text += text

print(len(raw_text))

# Splitting up the text into smaller chunks for indexing
text_splitter = CharacterTextSplitter(        
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 200, #striding over the text
    length_function = len,
)
texts = text_splitter.split_text(raw_text)

#import and normalise the data 
import re
# s is input text
def normalize_text(s, sep_token = " \n "):
    s = re.sub(r'\s+',  ' ', s).strip()
    s = re.sub(r". ,","",s)
    # remove all instances of multiple spaces
    s = s.replace("..",".")
    s = s.replace(". .",".")
    s = s.replace("\n", "")
    s = s.strip()
    return s

texts = list(map(normalize_text, texts))

#Generate embeddings 
from langchain.vectorstores import FAISS 
embeddings = OpenAIEmbeddings()
docsearch = FAISS.from_texts(texts, embeddings)
docsearch.embedding_function


#Create a Plain QA Chain 
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

chain = load_qa_chain(OpenAI(), chain_type="stuff")

query = "Is life short?"
docs = docsearch.similarity_search(query_02)
chain.run(input_documents=docs, question=query)
