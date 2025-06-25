from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline  # yeni paket
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationChain

def main():
    print("Bot başlatıldı. Çıkmak için 'çık' yazın.\n")

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

    memory = ConversationSummaryMemory(llm=llm)  # özetleme memory

    conversation = ConversationChain(llm=llm, memory=memory, verbose=False)

    while True:
        user_input = input("Sen: ")
        if user_input.lower() == "çık":
            print("Görüşürüz!")
            break
        if user_input.lower() == "özetle":
            summary = memory.load_memory_variables({}).get("history", "Özet bulunamadı.")
            print("Bot (Özet):", summary)
            continue
        response = conversation.predict(input=user_input)
        print("Bot:", response)

if __name__ == "__main__":
    main()
