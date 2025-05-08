import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

def main():
    client = Groq(
        api_key=os.getenv("GROQ_KEY")
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Write a poem on the beauty of mathematics."
            }
        ],
        model="llama-3.3-70b-versatile"
    )
    
    print(chat_completion.choices[0].message.content)