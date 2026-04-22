**Technical Task 2: Build a Retrieval-Augmented Chatbot (RAG)
Project: PoBot Expansion – AI Assistant for Migrant Support
Objective**
The goal of this task is to assess your ability to build a simple AI-powered question-answering
system using Retrieval-Augmented Generation (RAG). You will collect regulatory data,
preprocess it, and develop a basic chatbot that provides accurate information about Hong Kong
labor regulations.
**Task Description**
You are required to build a **basic chatbot system** that answers user questions related to **Hong
Kong labor and employment regulations** using a Retrieval-Augmented Generation (RAG)
pipeline.
**Requirements**

**1. Data Collection**
    ● Collect at least **5 official or reliable documents** related to Hong Kong labor regulations
       (e.g., employment rights, recruitment rules, migrant worker protections)
    ● Sources may include government websites, policy documents, or legal guidelines
    ● Store the collected data in a usable format (e.g., text, PDF, or HTML)
**2. Data Preprocessing**
Prepare the collected documents by:
    ● Extracting and cleaning text from raw sources
    ● Removing irrelevant content (e.g., navigation menus, headers/footers)
    ● Splitting the text into smaller chunks suitable for retrieval
    ● Ensuring consistency and readability of the text
**3. Embedding and Retrieval Setup**
    ● Convert text chunks into embeddings using any model (open-source or API-based)
    ● Store embeddings in a vector database (e.g., FAISS or similar)


```
● Implement a retrieval mechanism to fetch relevant chunks based on a user query
```
**4. RAG Pipeline Implementation**
    ● Integrate a language model (open-source or closed API) to generate answers
    ● Design a prompt that ensures responses are grounded in the retrieved context
    ● The system should:
       ○ Take a user question as input
       ○ Retrieve relevant document chunks
       ○ Generate a response based on the retrieved information
**5. Chatbot Interface**
    ● Create a simple testing function to run the pipeline (No UI needed)
    ● The chatbot should be able to answer questions such as:
       ○ “What are the rights of domestic workers in Hong Kong?”
       ○ “What are the rules for recruitment agencies?”
**6. Basic Evaluation**
    ● Provide at least **3–5 example queries**
    ● Show the chatbot’s responses
    ● Briefly describe one limitation or failure case (e.g., incorrect or incomplete answer)
**Expected Output**
    ● A Python script or Jupyter Notebook implementing the RAG pipeline
    ● A folder or file containing the collected and processed documents
    ● A working chatbot (CLI or simple UI)
    ● Sample queries and outputs
    ● A short summary (3–5 sentences) describing your approach and findings
**Evaluation Criteria**
    ● Quality and relevance of collected data
    ● Effectiveness of preprocessing and chunking
    ● Correct implementation of the RAG pipeline


● Clarity and usefulness of chatbot responses
● Code organization and documentation
**Bonus (Optional)**
● Include source citations in chatbot responses
● Support multilingual queries (e.g., English and Chinese)
● Add a simple confidence or fallback response for uncertain answers
● Use an agent-based approach or tool integration


