from IPython import get_ipython
from IPython.display import display_markdown

from groq import Groq

class ReflectionAgent:
    def __init__(
        self, model: str, api_key: str
    ):
        self.client = Groq(api_key=api_key)
        self.model = model
        self.generation_chat_history = None
        self.reflection_chat_history = None
    
    def generate(self, prompt, n_steps: int = 3, verbose: str = True):
        if self.generation_chat_history is None:
            self.generation_chat_history = [
                {
                    "role": "system",
                    "content": self.generation_system_prompt
                }
            ]
            
            self.generation_chat_history.append(
                {
                    "role": "user",
                    "content": prompt
                }
            )
        
        if self.reflection_chat_history is None:
            self.reflection_chat_history = [
                {
                    "role": "system",
                    "content": self.reflection_system_prompt
                }
            ]
        
        for step in range(n_steps):
            response = self.client.chat.completions.create(
                messages=self.generation_chat_history,
                model=self.model
            ).choices[0].message.content
            
            self.generation_chat_history.append(
                {
                    "role": "assistant",
                    "content": response
                }
            )
            
            self.reflection_chat_history.append(
                {
                    "role": "user",
                    "content": response
                }
            )
            
            critique = self.client.chat.completions.create(
                messages=self.reflection_chat_history,
                model=self.model
            ).choices[0].message.content
            
            self.generation_chat_history.append(
                {
                    "role": "user",
                    "content": critique
                }
            )
            
            if verbose:
                print(f"[bold green]Step: {step}\n")
                if get_ipython() is not None:
                    display_markdown(response, raw=True)
                    display_markdown(critique, raw=True)
                else:
                    print(response)
                    print("[blue]" + critique)
                    
        
        response = self.client.chat.completions.create(
            messages=self.generation_chat_history,
            model=self.model
        ).choices[0].message.content
        
        return response
    
    def set_generation_system_prompt(self, prompt: str):
        self.generation_system_prompt = prompt
        
    def set_reflection_system_prompt(self, prompt: str):
        self.reflection_system_prompt = prompt
        
    def reset(self):
        self.generation_chat_history = None
        self.reflection_chat_history = None