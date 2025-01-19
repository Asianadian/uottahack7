from langchain_community.chat_models import ChatOpenAI
from langchain.tools import Tool
from langchain.agents import initialize_agent
from llm import LLM
from google_calendar import GoogleCalendar
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

llm = LLM(OPENAI_API_KEY)
google_calendar = GoogleCalendar()

def create_events(prompt):
  print()
  ics_data = llm.create_event_prompt(prompt)
  added_events = google_calendar.create_events(ics_data)
  return added_events
  ...

def delete_events(prompt):
  print()
  events = google_calendar.get_events()
  event_ids_to_delete = llm.delete_event_prompt(prompt, events)
  deleted_events = google_calendar.delete_events(event_ids_to_delete)
  return deleted_events
  ...

create_event_tool = Tool(
  name='CreateEvents',
  func=create_events,
  return_direct=True,
  description='Use this for scheduling events'
)

delete_event_tool = Tool(
  name='DeleteEvents',
  func=delete_events,
  return_direct=True,
  description='Use this for deleting events'
)

tools = [create_event_tool, delete_event_tool]

class LangChainAgent:
  def __init__(self, api_key):
    os.environ["OPENAI_API_KEY"] = api_key
    self.llm = ChatOpenAI(temperature=0.1)
    self.agent = initialize_agent(tools, self.llm, agent="zero-shot-react-description", verbose=True)

  def run(self, prompt):
    a = self.agent.run(prompt)
    return a


  
