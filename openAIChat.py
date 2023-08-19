from langchain.tools import PythonREPLTool, Tool
import os
from dotenv import load_dotenv
from langchain import WikipediaAPIWrapper


from langchain.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain.agents import AgentType, initialize_agent
from langchain.agents import AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WikipediaLoader

load_dotenv()


class Config():
    """
    Contains the configuration of the LLM.
    """
    model = 'gpt-3.5-turbo'
    llm = ChatOpenAI(temperature=0, model=model)


def setup_agent() -> AgentExecutor:
   """
   Sets up the tools for a function based chain.
    We have here the following tools:
    - wikipedia
    - duduckgo
   """
   cfg = Config()
   duckduck_search = DuckDuckGoSearchAPIWrapper()
   wikipedia = WikipediaAPIWrapper()
   #wikipedia.lang="kr"
   #loader = WikipediaLoader(lang="kr",  query="");

 

   tools = [

      
        Tool(
            name="Wikipedia",
            func=wikipedia.run,
            description="useful when you need an answer about encyclopedic general knowledge"
           
        )
       
       
      ]

   grand_agent = initialize_agent(
        tools, 
        cfg.llm, 
        agent=AgentType.OPENAI_FUNCTIONS, 
        verbose=True
   )
    
   result=grand_agent.run("뉴진스")
   print(result)

   """
   agent_kwargs, memory = setup_memory()

   return initialize_agent(
         tools, 
         cfg.llm, 
         agent=AgentType.OPENAI_FUNCTIONS, 
         verbose=False, 
         agent_kwargs=agent_kwargs,
         memory=memory
   )
   """


if __name__ == '__main__':
   print('open ai chat')
   print(os.environ['OPENAI_API_KEY'])
   setup_agent()