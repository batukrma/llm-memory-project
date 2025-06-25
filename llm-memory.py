from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline  
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationChain

def main():
    print("Bot started. Type 'exit' to quit.\n")

    model_name = "distilgpt2"

    pipe = pipeline(
        "text-generation",
        model=model_name,
        max_new_tokens=50,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.15,
        truncation=True
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    memory = ConversationSummaryMemory(llm=llm)  

    conversation = ConversationChain(llm=llm, memory=memory, verbose=False)

    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        if user_input.lower() == "summary":
            summary = memory.load_memory_variables({}).get("history", "No summary found.")
            print("Bot (Summary):", summary)
            continue
        response = conversation.predict(input=user_input)
        print("Bot:", response)

if __name__ == "__main__":
    main()
