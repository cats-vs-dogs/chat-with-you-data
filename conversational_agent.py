from langchain.llms import OpenAI
# from langchain.chat_models import ChatOpenAI
from langchain import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.agents import initialize_agent, Tool
# from langchain.agents import load_tools
from langchain.memory import ConversationBufferMemory

from rw_formula import RW_Calc2


import os
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


import anvil.server
anvil.server.connect("server_KNICBILM63PHH7ZNOJ45HGXU-MBANGIRAJW2TWQDC")


llm = OpenAI(temperature=0)
# llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
db = SQLDatabase.from_uri("sqlite:///./rwa_data.db")
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
tools = [
    Tool(
        name="risk weight calculator",
        func=RW_Calc2,
        description=
        """calculates a loan risk weight give the argulamts: 
        Segment - Possible values are 'Bank', 'Corporate', and 'Retail'), 
        PD - Probability of Default
        LGD - Loss Given Default
        m - Remaining maturity of the loan in years
        Large_Fin - If 'Y' the client is a Flag for Large Financial Institution, otherwise 'N'
        size - size of the client in MEUR, usually this is the client's turnover
        mortgage - If 'Y' the exposure is a mortgage loan, otherwise 'N'
        revolving - If 'Y' the exposure is a revolving loan, otherwise 'N'
        """
    ),     
    Tool(
        name="rwa_db",
        func=db_chain.run,
        description="useful for quering a database"
    )
]

memory = ConversationBufferMemory(memory_key="chat_history")

agent = initialize_agent(
    agent='conversational-react-description', 
    tools=tools, 
    llm=llm,
    verbose=False,
    # max_iterations=3,
    memory=memory,
    handle_parsing_errors=True
)


# agent.run(query)


######################

# @anvil.server.callable
# def ask_buttun_click(self, **event_args):
#    result = agent.run(self.question.text)
#    self.answer.text = result


# @anvil.server.callable
# def ask_buttun_click(self, **event_args):
# #    query = self.question.text
#    result = anvil.server.call('agent', self.question.text)
#    self.answer.text = result


# NB
# define the agent in a function and then call it below as argument


@anvil.server.callable
def give_answer(query):
   result = agent.run(query)
   return result




# def ask_buttun_click(self, **event_args):
# #    query = self.question.text
#    result = anvil.server.call('agent', self.question.text)
#    self.answer.text = result



# def categorise_button_click(self, **event_args):
#     """This method is called when the button is clicked"""
#     # Call the server function and pass it the iris measurements
#     iris_category = anvil.server.call('predict_iris', 
#                                 self.sepal_length.text,
#                                 self.sepal_width.text,
#                                 self.petal_length.text,
#                                 self.petal_width.text)
#     # If a category is returned set our species 
#     if iris_category:
#       self.species_label.visible = True
#       self.species_label.text = "The species is " + iris_category.capitalize()


######################



anvil.server.wait_forever()


####################



# # notes:
# # Create the UI window
# root = tk.Tk()
# root.title("Chat with your Tabular Data")

# # Create the text entry widget
# entry = ttk.Entry(root, font=("Arial", 14))
# entry.pack(padx=20, pady=20, fill=tk.X)

# # Create the button callback
# def on_click():
#     # Get the query text from the entry widget
#     query = entry.get()

#     # Run the query using the agent executor
#     result = agent_executor.run(query)

#     # Display the result in the text widget
#     text.delete("1.0", tk.END)
#     text.insert(tk.END, result)

# # Create the button widget
# button = ttk.Button(root, text="Chat", command=on_click)
# button.pack(padx=20, pady=20)

# # Create the text widget to display the result
# text = tk.Text(root, height=10, width=60, font=("Arial", 14))
# text.pack(padx=20, pady=20)

# # Start the UI event loop
# root.mainloop()