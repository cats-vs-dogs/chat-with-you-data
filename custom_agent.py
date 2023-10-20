from langchain.chat_models import ChatOpenAI
from langchain import SQLDatabase
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser, create_sql_agent, load_tools 
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.prompts import StringPromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.tools import Tool
# from langchain.tools import Tool, BaseTool
from langchain.schema import AgentAction, AgentFinish, OutputParserException
from pydantic import BaseModel, Field
from scipy.stats import norm
from typing import Union
# from typing import List, Union
from dotenv import load_dotenv
import anvil.server
import math, re, os

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

anvil.server.connect("server_KNICBILM63PHH7ZNOJ45HGXU-MBANGIRAJW2TWQDC") # update this

llm = ChatOpenAI(model_name='gpt-4',temperature=0)
# llm = ChatOpenAI(model_name="gpt-3.5-turbo-0613",temperature=0)
db = SQLDatabase.from_uri("sqlite:///./portfolio_data.db",
                         include_tables=['corporate_portfolio'],
                         sample_rows_in_table_info=2)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
sql_agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
) 

def rw_corp(Input_String: str) -> str:

    """ 
    calculates the risk weight from input parameters 
    PD, LGD, MATURITY, F_LARGE_FIN and SIZE which are fed into the formula in this particular order 
    """    
    PD_s, LGD_s, MATURITY_s, SIZE_s, F_LARGE_FIN_s = Input_String.split(",")

    PD = float(eval(PD_s.split(":")[1].strip()))
    LGD = float(eval(LGD_s.split(":")[1].strip()))
    MATURITY = float(eval(MATURITY_s.split(":")[1].strip()))
    SIZE = float(eval(SIZE_s.split(":")[1].strip()))
    F_LARGE_FIN = (F_LARGE_FIN_s.split(":")[1].strip())
       
    pd_final = max(0.0003, PD)
    size_final = max(5, min(SIZE, 50))    
    r0 = (0.12 * (1.0 - math.exp(-50.0 * pd_final)) / (1.0 - math.exp(-50.0))) + \
        (0.24 * (1.0 - (1 - math.exp(-50.0 * pd_final)) / (1.0 - math.exp(-50.0)))) - \
        (0.04 * (1.0 - (size_final - 5.0) / 45.0))
    if F_LARGE_FIN == 'Y':
        r = r0 * 1.25
    else:
        r = r0
    b = (0.11852 - 0.05478 * math.log(pd_final)) ** 2
    ma = ((1 - 1.5 * b) ** -1) * (1 + (MATURITY - 2.5) * b)    
    rw = ((LGD * norm.cdf((1 - r) ** -0.5 * norm.ppf(pd_final) + (r / (1 - r)) ** 0.5 * norm.ppf(0.999)) - pd_final * LGD) * ma) * 12.5 * 1.06
    return rw  

class RWInput(BaseModel):
    Input_String: str = Field(description='This is a string that contains values for the input parameters \
                              PD, LGD, MATURITY, SIZE and F_LARGE_FIN which are fed into the formula in this particular order ')
    
RWTool = Tool.from_function(
    func=rw_corp,
    name="RWTool",
    description="""
    This is a custom tool that calculates the risk weight from a set of input parameters:
        PD - Probability of Default,
        LGD - Loss Given Default,
        MATURITY - Remaining maturity of the loan in years,
        SIZE - The size of the client in MEUR, usually this is the client's turnover, 
        F_LARGE_FIN - If 'Y' the client is a Large Financial Institution        
        """,
    args_schema=RWInput
)

extra_tools = load_tools(
    ['llm-math'],
    llm=llm
)
extra_tools.append(RWTool)
tools = toolkit.get_tools() + list(extra_tools)

template = """
You are an agent designed to interact with a SQL database and use the information extracted from that database to perform additional calcuations if necessay.
Given an input question, create a syntactically correct sqlite query to run, then look at the results of the query and return the answer. 
If the question requires it, use the query result as input to additional calculations that will return the final answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.
DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
You have access to tools for interacting with the database and calculating the final result with data extracted from the database.
Only use the below tools. Only use the information returned by the below tools to construct your final answer.
If the question does not seem related to the database, just return "I don't know" as the answer.

Use the following tools:
sql_db_query: Input to this tool is a detailed and correct SQL query, output is a result from the database. If the query is not correct, an error message will be returned. If an error is returned, rewrite the query, check the query, and try again. If you encounter an issue with Unknown column 'xxxx' in 'field list', use sql_db_schema to query the correct table fields.
sql_db_schema: Input to this tool is a comma-separated list of tables, output is the schema and sample rows for those tables. Be sure that the tables actually exist by calling sql_db_list_tables first! Example Input: 'table1, table2, table3'
sql_db_list_tables: Input is an empty string, output is a comma separated list of tables in the database.
sql_db_query_checker: Use this tool to double check if your query is correct before executing it. Always use this tool before executing a query with sql_db_query!
Calculator: Useful for when you need to answer questions about math.
RWTool: 
    This is a custom tool that calculates the risk weight from a set of input parameters:
        PD - Probability of Default,
        LGD - Loss Given Default,
        MATURITY - Remaining maturity of the loan in years,
        SIZE - The size of the client in MEUR, usually this is the client's turnover, 
        F_LARGE_FIN - If 'Y' the client is a Large Financial Institution               

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [sql_db_query, sql_db_schema, sql_db_list_tables, sql_db_query_checker, Calculator, RWTool]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Previous conversation history:
{history}

Question: {input}
Thought: I should look at the tables in the database to see what I can query. Then I should query the schema of the most relevant tables. Based on the question I may need to use the result of the query to carry out additional calcualtion.
{agent_scratchpad}
"""

class CustomPromptTemplate(StringPromptTemplate):
    template: str
    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        kwargs["agent_scratchpad"] = thoughts
        return self.template.format(**kwargs)
    

prompt = CustomPromptTemplate(
    template=template,
    input_variables=["input", "intermediate_steps", "history"]
)

class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise OutputParserException(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)    
output_parser = CustomOutputParser()


llm_chain = LLMChain(llm=llm, prompt=prompt)
tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\nObservation:"],
    allowed_tools=tool_names
)
memory=ConversationBufferWindowMemory(k=4)
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory)

# output = agent_executor.run("What are total RWA and EAD for reporting date 31-12-2022?")

@anvil.server.callable
def give_answer(query):
   result = agent_executor.run(query)
   return result

anvil.server.wait_forever()


# previous
# llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
# db = SQLDatabase.from_uri("sqlite:///./rwa_data.db")
# db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

# extract_tools = [    
#     Tool(
#         name="rwa_db",
#         func=db_chain.run,
#         description="useful for quering a database"
#     )
# ]

# extract_agent = initialize_agent(
#     tools=extract_tools, 
#     llm=llm, 
#     agent="zero-shot-react-description", 
#     verbose=False, 
#     handle_parsing_errors=True
# )

# extract_and_calc_tools = [
#     Tool(
#         name="risk weight calculator",
#         func=RW_Calc,
#         description=
#         """calculates a loan risk weight give the argulamts: 
#         Segment - Possible values are 'Bank', 'Corporate', and 'Retail'), 
#         PD - Probability of Default
#         LGD - Loss Given Default
#         m - Remaining maturity of the loan in years
#         Large_Fin - If 'Y' the client is a Flag for Large Financial Institution, otherwise 'N'
#         size - size of the client in MEUR, usually this is the client's turnover
#         mortgage - If 'Y' the exposure is a mortgage loan, otherwise 'N'
#         revolving - If 'Y' the exposure is a revolving loan, otherwise 'N'
#         """
#     ),    
#     Tool(
#         name="rwa_db",
#         func=db_chain.run,
#         description="useful for quering a database"
#     )
# ]

# extract_and_calc_agent = initialize_agent(
#     tools=extract_and_calc_tools, 
#     llm=llm, 
#     agent="zero-shot-react-description", 
#     verbose=False, 
#     handle_parsing_errors=True
# )

# @anvil.server.callable
# def give_answer_extract(query):
#    result = extract_agent.run(query)
#    return result

# @anvil.server.callable
# def give_answer_extract_and_calc(query):
#    result = extract_and_calc_agent.run(query)
#    return result

# anvil.server.wait_forever()

# previous