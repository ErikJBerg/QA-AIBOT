# Chatbot for Card Blocking and Account Opening

This repository contains the code for a service bot that can handle user requests related to:
- card blocking
- opening an account
- answering frequently asked questions (FAQs). 
 
The bot utilizes Language Models (LLMs) from openai to provide accurate and relevant responses.

## Getting Started

To run the code and interact with the chatbot, please follow the instructions below:

### Prerequisites

Make sure you have the following dependencies installed:

- Python 3.x
- All the required libraries mentioned in the `requirements.txt` file.

You can install the dependencies by running the following command:

```bash
pip install -r requirements.txt
```

### Running the Code

1. Clone this repository to your local machine or download the ZIP file and extract its contents.

2. Navigate to the `working_example` folder.

3. Open a terminal or command prompt in the `working_example` folder.

4. Run the following command to start the chatbot:

   ```bash
   python qa_intent_bot.py
   ```

5. Once the chatbot is running, you can interact with it by typing your queries and pressing Enter.

6. To exit the chatbot, type `exit` and press Enter.

## Usage

The chatbot is designed to handle user requests on two main topics: card blocking and opening an account. 
Additionally, it can answer frequently asked questions from the provided FAQ documents.

It is still a work in progress and can be improved in many ways, if there was more time to spend on it. 
For example, it can be extended to contain more comprehensive intent recognition or use better similarity search to switch between information retrieval chains.
Usage of agents and connecting to a real database, even if sqlite would also make it more user-friendly.
Except persistence implementing better flow manager switching between intent chains/agents would also improve user experience.

### Happy Path Scenarios

To test the chatbot's functionality, you can try the following examples:

1. **Card Blocking**
2. **Account Opening**
3. **FAQ**

Specific examples for each scenario both suffesful and as of yet not succcesful can be found in files:
- happy_paths.txt
- failed_paths.txt

---

### Way of thought
To see my way of thoughts when trying to solve the usecase, you can check jupyter notebooks in this repository.


