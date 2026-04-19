from autogen_agentchat.agents import CodeExecutorAgent
from autogen_agentchat.messages import TextMessage
import asyncio
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_core import CancellationToken
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelInfo
from dotenv import load_dotenv
import os
from autogen_agentchat.ui import Console
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat




load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
model_name = os.environ.get("GEMINI_MODEL")

model_client = OpenAIChatCompletionClient(    
    model=model_name,
    api_key=GOOGLE_API_KEY,
    api_type = "google",
    model_info=ModelInfo(
        vision=True,
        function_calling=True,
        temperature=0.7,
        json_output=True,
        structured_output=True,
        family="unknown",

    )
)


async def main():
    local_executor = LocalCommandLineCodeExecutor(
        work_dir="/tmp",   # or "F:/msautogen/tmp" on Windows
        timeout=120
    )

    code_executor_agent = CodeExecutorAgent(
        name="CodeExecutorAgent",
        code_executor=local_executor
    )

#     task = TextMessage(
#         content= '''Here is some code
# ```python
# print('Hello world')
# ```
#     ''',
#         source = "user")
#     result = await code_executor_agent.on_messages(
#     [task],
#     cancellation_token=CancellationToken()
# )

    problem_solver_agent = AssistantAgent(
        name = "DSA_Problem_Solver_Agent",
        description = "An aget that solves DSA problems",
        model_client = model_client,
        system_message = '''
    You are a problem solver agent that is an expert in solving DSA Problems. 
    You will be working with code executor agent to execute code.
    You will be given a taskand you should 
    1. Write code to solve the task. Your code should only be in python
    At the beginning of your resposnse you have to specify your plan to solve the task.
    You should give the code in a code block.(Python)
    You should write code in a one code block at a time and pass it to code executor agent to execute it.
    Make sure that we have atleast 3 test cases for the code you write.
    Once the code is executed and if the same has been done succesfullyyou have the results.
    You should explain the code execution result.

    In the end when the code is executed succesfully you have to say "STOP" to stop the conversation.


    '''
    )

    termination_condition = TextMentionTermination("STOP")
    team = RoundRobinGroupChat(
        participants = [problem_solver_agent, code_executor_agent],
        termination_condition = termination_condition,
        max_turns=10,
    )

    try:
        task = "Write a python code to add two numbers"
        async for message in team.run_stream(task=task):
            print('=='* 20)
            print(message.source, ":", message)
            print('=='* 20)
 

    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        print("Execution completed")


if __name__ == "__main__":
    asyncio.run(main())