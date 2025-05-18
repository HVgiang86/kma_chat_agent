import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from agent.state import MyAgentState, InitThinkingOutput, RePlanningOutput, DoTaskOutput
from llm.config import get_gemini_llm
from rag import create_rag_tool
from score import get_student_scores, get_student_info, calculate_average_scores

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define tools
score_tool = get_student_scores
student_info_tool = get_student_info
rag_tool = create_rag_tool()
calculator_tool = calculate_average_scores

# Get all tools
tools = [score_tool, student_info_tool, calculator_tool, rag_tool]

INIT_THINKING_NODE = "init_thinking"
RE_PLAIN_NODE = "re_plain"
DO_TASK_NODE = "do_task"
TOOL_NODE = "tool"

react_prompt = """
You are KBot - a helpful AI assistant for KMA (Academy of Cryptography Techniques) students. This is planning phase.
Your job is to help students with their questions about KMA regulations, their information, and their scores.

You have access to the following tools:
{tool_descriptions}

- search_kma_regulations: Search for information in KMA's regulations, rules, and policies - USE THIS TOOL FIRST for any questions about KMA rules, requirements, policies, or procedures
- get_student_scores: Get student's scores with filtering options (semester must be in format ki1-2024-2025, k2-2024-2025, etc.)
- get_student_info: Get student's information (name, class)
- calculate_average_scores: Calculate average scores for a student

Follow this process:
1. Understand the student's question and identify the core task.
2. Check if you have all necessary information to use a tool (e.g., `student_code` is often required).
3. If essential information for a tool is missing from the user's query and not available in your current memory/state from previous turns (e.g. if student_code was provided in a previous message), you should respond that you cannot perform the action without the specific missing information. Do NOT make up information.
4. If you have the necessary information, decide which tool(s) are most appropriate.
    - For questions about regulations, requirements, or policies, use `search_kma_regulations`.
    - For questions about a student's specific scores, if `student_code` is available, use `get_student_scores`.
    - For questions about a student's general information (name, class), if `student_code` is available, use `get_student_info`.
    - For questions about averages or GPAs, if `student_code` is available, use `calculate_average_scores`.
5. Execute the chosen tool(s).
6. Review the tool's output.
7. Formulate a final answer to the student based on the tool's output or if no tool was needed.
8. If a tool returns an error or indicates missing information that you cannot obtain, clearly state this in your response.

Important considerations: 
- If a tool requires a `student_code` and it's not provided in the initial query, state that you need it. For example: "To get student scores, I need the student's code. Please provide it."
- Present scores or information from regulations in a readable format.
- If `search_kma_regulations` is used, cite sources if appropriate and available from the tool's output.
- Provide a direct final answer once all necessary information is gathered and processed. Do not ask clarifying questions if the original query is clear enough to attempt a tool call or provide a direct answer.
- If you determine that no tool is needed, or you cannot proceed due to missing information that was not provided, directly answer the user.
"""

thinking_prompt = """
** CONTEXT **

You are do-task agent of KBot - a helpful AI assistant for KMA (Academy of Cryptography Techniques) students.
Your job is to help students with their questions about KMA regulations, their information, their schedule and their scores or some thing else.

Your user is a student at KMA (Academy of Cryptography Techniques).

They might greet you or ask a question or request information.
Your task is to THINK ABOUT HOW to answer to the students.

** TASK **

Based on the student's chat, create a detailed plan with step-by-step instructions on how to respond.

You have access to the following tools:
{tool_descriptions}

Your plan should include:
1. Identify the core task or question.
2. Determine if you have all the necessary information to answer the question.
3. If not, identify what information is missing and how to obtain it.
4. Decide which tools are appropriate to use based on the question and available information.
5. Specify the parameters needed for each tool.
6. Outline the logical sequence of actions to take.

Break down your thinking process into clear steps. Be specific about what information is available and what might be missing.
DO NOT try to answer the question directly. Just plan the approach. Output a list of steps prefixed with "STEP 1:", "STEP 2:", etc.

Your output format should be in json format that can extract:

field message, 
field thinking_steps is array of thinking element json that each format as :
     index : ,
     step : ,

** EXAMPLE **
For example, if the student asks about their scores:
- Identify the student's code and semester.
- Check if the student code is provided in the chat.
- If the student code is missing, ask the student for it.
- If the semester is missing, maybe user want to all scores, so you can ignore it.
- If both are available, prepare to call the `get_student_scores` tool with the provided parameters.
- If the tool call is successful, format the output and provide a clear answer to the student.

If the student asks about can he get a scholarship:
- Get KMA's regulations about scholarships.
- Based on the regulations, check if the student is eligible for a scholarship.
- If the student is eligible, provide information about the scholarship process.
- If need to get student scores, check if the student code is provided.
- If the student code is missing, ask the student for it.

** IMPORTANT **
- DO NOT answer the question directly.
- DO NOT make up information.
- DO NOT ask clarifying questions if the original query is clear enough to attempt a tool call or provide a direct answer.
- DO NOT provide information that is not relevant to the question.
- DO NOT provide information that is not available in the tools or your knowledge.
- If student's question is inconvenient like just greeting, you can ignore planning and just answer it.

Consider:
1. What information do I need to answer this question?
2. Which tools would be appropriate to use?
3. What specific parameters do I need for each tool?
4. Is there any missing information that I need to ask the student for?
5. What is the logical sequence of actions I should take?
...
"""

replanning_prompt = """
** CONTEXT **
You are re-planning agent of KBot - a helpful AI assistant for KMA (Academy of Cryptography Techniques) students.
KBot job is to help students with their questions about KMA regulations, their information, their schedule and their scores or some thing else.
Your user is a student at KMA (Academy of Cryptography Techniques).

You has provided a plan of steps to answer the user's question and information each step gathered.
Your task is to THINK AND DECIDE: NEED TO RE-PLAN the steps based on new information that has been gathered or NOT.

You are KBot - a helpful AI assistant for KMA (Academy of Cryptography Techniques) students.
You are in the process of REPLANNING your approach based on new information you've gathered.

** TASK **
Based on the student's original question and the NEW INFORMATION you've gathered from your previous actions,
DECIDE if you need to re-plan your approach or not.
-> IF NO NEED to re-plan: response that do not need.
-> IF NEED to re-plan: create a REVISED plan with step-by-step instructions on how to proceed.

** IMPORTANT **
- DO NOT answer the question directly.
- DO NOT make up information.
- DO NOT ask clarifying questions if the original query is clear enough to attempt a tool call or provide a direct answer.
- NEW PLAN (if has) should not include any steps that have already been completed.
- NEW PLAN is continuation of the previous plan, so it should not repeat any steps that have already been completed.
- NEW PLAN should be specific about what information you now have and what might still be missing.
- Focus on how the NEW INFORMATION changes your approach
- Be specific about what NEXT STEPS to take
- DO NOT try to answer the question yet - just plan the remaining steps

** FORMAT **
You should respond all plan include complete - previous steps and new plan
Your output format should be in json format that can extract:

field need_replan: boolean
field message: string, 
field thinking_steps is array of thinking element json that each format as :
     index : ,
     step : ,
     step_result: ,
     is_done: true/false

** DOCUMENT **
You have access to the following tools:
{tool_descriptions}

Your previous plan was:
{previous_thinking}

You have already performed some actions and gathered new information. The most recent relevant information steps gatherd is:
{recent_information}

** EXAMPLE **
For example, if the student asked about eligibility for a scholarship. If previous steps is
- Retrieve KMA's regulations about scholarships - done - RESULT: GPA >= 3.0 at the end of the semester
- Get information about the student - not done yet 

You see that need to re-plan because you need to get the student scors at that semester to check if the student is eligible for a scholarship.
=> Response Re-plan continue steps as:
- Get the student scores at the end of the semester

"""

do_task_prompt = """
** CONTEXT **
You are do-task agent of KBot - a helpful AI assistant for KMA (Academy of Cryptography Techniques) students.
KBot job is to help students with their questions about KMA regulations, their information, their schedule and their scores or some thing else.
Your user is a student at KMA (Academy of Cryptography Techniques).

You has provided a plan of steps to answer the user's question and information each step gathered.
Your task is take action to do task based on the plan and the information that has been gathered.
TAKE ACTION!

** TASK **
Based on the student's original question and the ACTION PLAN and  the INFORMATION you've gathered from your previous actions,
take actions to do task and response to user.
You should do single step.

** PROCESS TO DO TASK **
Follow this process:
1. Understand the student's question and tasks in action plan. Do the not-completed tasks.
2. Check if you have all necessary information to use a tool (e.g., `student_code` is often required).
3. If essential information for a tool is missing from the user's query and not available in your current memory/state from previous turns (e.g. if student_code was provided in a previous message), you should respond that you cannot perform the action without the specific missing information. Do NOT make up information.
4. If you have the necessary information, decide which tool(s) are most appropriate.
    - For questions about regulations, requirements, or policies, use `search_kma_regulations`.
    - For questions about a student's specific scores, if `student_code` is available, use `get_student_scores`.
    - For questions about a student's general information (name, class), if `student_code` is available, use `get_student_info`.
    - For questions about averages or GPAs, if `student_code` is available, use `calculate_average_scores`.
5. Execute the chosen tool(s).
6. Review the tool's output.
7. Formulate a final answer to the student based on the tool's output or if no tool was needed.
8. If a tool returns an error or indicates missing information that you cannot obtain, clearly state this in your response.

** IMPORTANT **
Important considerations: 
- If a tool requires a `student_code` and it's not provided in the initial query, state that you need it. For example: "To get student scores, I need the student's code. Please provide it."
- Present scores or information from regulations in a readable format.
- If `search_kma_regulations` is used, cite sources if appropriate and available from the tool's output.
- Provide a direct final answer once all necessary information is gathered and processed. Do not ask clarifying questions if the original query is clear enough to attempt a tool call or provide a direct answer.
- If you determine that no tool is needed, or you cannot proceed due to missing information that was not provided, directly answer the user.

** FORMAT **
You should respond all plan include complete - previous steps and new plan
Your output format should be in json format that can extract:

field message: string, 
field result: string,
field completed_steps is array of index number of steps that you did in this session.
field has_final_result indicate if you have final result or not

** DOCUMENT **
You have access to the following tools:
{tool_descriptions}

Your action plan has following steps:
{steps}

What you got:
{recent_information}

"""


def get_tool_descriptions(tools_list: list) -> str:
    return "\n".join([
        f"- {tool.name}: {tool.description} (args: {tool.args_schema.schema()['properties'].keys() if tool.args_schema else 'None'})"
        for tool in tools_list])


async def init_thinking_agent(state: MyAgentState) -> MyAgentState:
    """
    Initialize the thinking agent with the initial state
    """
    logger.info("--- [NODE] -> Init Thinking ---")

    prompt = ChatPromptTemplate.from_messages(
        [("system", thinking_prompt.format(tool_descriptions=get_tool_descriptions(tools))),
         MessagesPlaceholder(variable_name="messages"), ])

    # Use the LLM to generate thinking steps
    llm = get_gemini_llm().with_structured_output(InitThinkingOutput)
    chain = prompt | llm

    logger.info("--- Input STATE ---")
    logger.info(state)

    try:
        thinking_response = chain.invoke({"messages": state["messages"]})

        logger.info(f"--- RAW RESPONSE ---")
        logger.info(thinking_response)

        # Extract steps from the thinking response
        if isinstance(thinking_response, InitThinkingOutput):
            messages = thinking_response.message
            steps = thinking_response.thinking_steps
        else:
            steps = [{"index": 1, "step": "Cannot plan the steps. Please answer that you don't know how to do it.",
                      "step_result": "", "is_done": False}]

        # Log the steps

        logger.info(f"--- THINKING AGENT: Steps generated ---")

        logger.info(f"--- RAW ---")
        logger.info(f"---Messages ---")
        logger.info(f"{messages}")
        logger.info(f"--- Thinking Steps ---")
        for step in steps:
            logger.info(step)

        new_state = {"messages": state["messages"] + [AIMessage(content=messages)], "thinking_steps": steps,
                     "finish": False, "previous_thinking": [], }

        ## Log the steps
        logger.info(f"--- THINKING AGENT: Steps ---")
        logger.info(f"--- New State ---")
        logger.info(f"--- Messages ---")
        logger.info(f"{new_state['messages']}")
        logger.info(f"--- Thinking Steps ---")
        for step in new_state['thinking_steps']:
            logger.info(step)

        logger.info(f"\n--- RETURN ---")

        logger.info(new_state)

        logger.info("--- [NODE] -> Init Thinking -> *END* ---\n\n")

        # Return the new state
        return new_state

    except Exception as e:
        logger.error(f"Error in thinking agent: {e}")
        steps = [
            {"index": 1, "step": "Has error in planning the steps. Please answer that you don't know how to do it.",
             "step_result": "", "is_done": False}]
        return_value = {"messages": state["messages"], "thinking_steps": steps, "previous_thinking": [],
                        "finish": False, }
        # ns = MyAgentState(**return_value)
        return return_value


async def replan_agent(state: MyAgentState) -> MyAgentState:
    """
    This agent will review the previous steps and the new information to decide need to replan or not.
    """
    logger.info("--- [NODE] -> Re-Planning AGENT ---")

    ## Get Previous thinking steps
    logger.info("--- THINKING AGENT: Previous thinking steps SETUP ---")
    previous_thinking_steps = state.get("previous_thinking", [])
    previous_step = []
    if not previous_thinking_steps:
        previous_thinking_steps = state.get("thinking_steps", [])

    if previous_thinking_steps:
        # If there are previous thinking steps, use them to inform the new thinking
        for step in previous_thinking_steps:
            logger.info(step)
            previous_step.append(step)

    ## Get recent information
    logger.info("--- THINKING AGENT: Recent information SETUP ---")
    recent_info = []
    if previous_thinking_steps:
        # If there are previous thinking steps, use them to inform the new thinking
        for step in previous_thinking_steps:
            if step.step_result:
                recent_info.append(step.step_result)

    recent_info_str = "\n".join(recent_info)
    logger.info(f"--- Recent information ---")
    logger.info(f"{recent_info_str}")

    # Create a prompt for replanning
    prompt = ChatPromptTemplate.from_messages([("system",
                                                replanning_prompt.format(tool_descriptions=get_tool_descriptions(tools),
                                                                         previous_thinking="\n".join(previous_step),
                                                                         recent_information=recent_info_str)),
                                               MessagesPlaceholder(variable_name="messages"), ])

    try:
        # Use the LLM to generate thinking steps
        llm = get_gemini_llm().with_structured_output(RePlanningOutput)

        logger.info("--- Input STATE ---")
        logger.info(state)

        thinking_response = llm.invoke([{"role": "user", "content": prompt}])

        logger.info(f"--- RAW RESPONSE ---")
        logger.info(thinking_response)

        steps = state['thinking_steps']
        previous_step = state['previous_thinking']

        # Extract steps from the thinking response
        if isinstance(thinking_response, RePlanningOutput):
            messages = thinking_response.message
            need_replan = thinking_response.need_replan

            if need_replan:
                new_steps = thinking_response.thinking_steps
                steps = steps + new_steps
                previous_step = state['thinking_steps']
        else:
            need_replan = False
            messages = "Cannot plan the steps. Please answer that you don't know how to do it."

        # Log the steps
        logger.info(f"--- Re-Plan AGENT: Steps generated ---")
        logger.info(f"--- RAW ---")
        logger.info(f"---Messages ---")
        logger.info(f"{messages}")
        logger.info(f"--- Thinking Steps ---")
        for step in steps:
            logger.info(step)
        new_state = {"messages": state["messages"] + [AIMessage(content=messages)], "thinking_steps": steps,
                     "previous_thinking": previous_step, "finish": False}

        return new_state

    except Exception as e:
        logger.error(f"Error in thinking agent: {e}")
        steps = [
            {"index": 1, "step": "Has error in planning the steps. Please answer that you don't know how to do it.",
             "step_result": "", "is_done": False}]
        return {"messages": state["messages"], "thinking_steps": steps, "previous_thinking": [], "finish": False, }


async def do_task(state: MyAgentState) -> MyAgentState:
    logger.info("--- [NODE] AGENT: Do Task with Tools ---")
    logger.info("--- INPUT STATE ---")
    logger.info(state)

    ## Get Previous thinking steps
    logger.info("--- DO TASK AGENT: Previous thinking steps SETUP ---")

    previous_thinking_steps = []
    if state.get("previous_thinking", []):
        previous_thinking_steps = state["previous_thinking"]

    if not previous_thinking_steps:
        if state.get("thinking_steps", []):
            previous_thinking_steps = state["thinking_steps"]

    previous_step = []
    logger.info(previous_thinking_steps)
    if previous_thinking_steps:
        # If there are previous thinking steps, use them to inform the new thinking
        logger.info("\n\n--- RAW ---")
        logger.info(previous_thinking_steps)
        for step in previous_thinking_steps:
            logger.info(step)
            previous_step.append(
                f"STEP {step.index}: {step.step} -> RESULT: {step.step_result} -> IS DONE?: {step.is_done}")

    logger.info("___HERE is previous steps:___")
    logger.info(previous_step)

    ## Get recent information
    logger.info("--- THINKING AGENT: Recent information SETUP ---")
    recent_info = []
    if previous_thinking_steps:
        logger.info(previous_thinking_steps)
        # If there are previous thinking steps, use them to inform the new thinking
        for step in previous_thinking_steps:
            logger.info("\n--- Thinking Steps ---")
            logger.info(step)
            if step.step_result:
                recent_info.append(step.step_result)

    recent_info_str = "\n".join(recent_info)
    logger.info(f"--- Recent information ---")
    logger.info(f"{recent_info_str}")

    # Create a prompt for replanning
    prompt = ChatPromptTemplate.from_messages([("system",
                                                do_task_prompt.format(tool_descriptions=get_tool_descriptions(tools),
                                                                      steps="\n".join(previous_step),
                                                                      recent_information=recent_info_str)),
                                               MessagesPlaceholder(variable_name="messages"), ])

    try:

        # Use the LLM to generate thinking steps
        llm = get_gemini_llm().bind_tools(tools)
        chain = prompt | llm

        logger.info("--- Input STATE ---")
        logger.info(state)

        task_result = chain.invoke({"messages": state["messages"]})

        logger.info(f"--- RAW RESPONSE ---")
        logger.info(task_result)

        # Extract steps from the thinking response
        if isinstance(task_result, DoTaskOutput):
            messages = task_result.message
            completed_steps = task_result.completed_steps
            has_final_result = task_result.has_final_result

            for step_index in completed_steps:
                for step in state['thinking_steps']:
                    if step.index == step_index:
                        step.is_done = True

            # Log the steps
            logger.info(f"--- DO TASK AGENT: Steps generated ---")
            logger.info(f"--- RAW ---")
            logger.info(f"---Messages ---")
            logger.info(f"{messages}")
            logger.info(f"--- Thinking Steps ---")
            for step in state['thinking_steps']:
                logger.info(step)

            new_state = {"messages": state["messages"] + [AIMessage(content=messages)],
                         "thinking_steps": state['thinking_steps'], "previous_thinking": previous_step,
                         "finish": has_final_result}
        else:
            new_state = {"messages": state["messages"] + [
                AIMessage(content="Cannot plan the steps. Please answer that you don't know how to do it.")],
                         "thinking_steps": state['thinking_steps'], "previous_thinking": previous_step,
                         "finish": False, }

        return new_state

    except Exception as e:
        logger.error(f"Error invoking LLM: {e}")
        error_message = AIMessage(content=f"An error occurred with the LLM: {e}")
        return {"messages": state["messages"] + [error_message], "thinking_steps": state['thinking_steps'],
                "previous_thinking": previous_step, "finish": False, }


def should_end(state: MyAgentState):
    logger.info("--- AGENT: Deciding next step for tool execution ---")

    # Check if replanning is needed first
    if state.get("finish", False):
        logger.info("--- AGENT: Replanning needed, returning to thinking ---")
        return END

    last_message = state['messages'][-1] if state['messages'] else None
    if not last_message:
        return END

    if isinstance(last_message, AIMessage) and hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return TOOL_NODE

    return RE_PLAIN_NODE


tool_node = ToolNode(tools)


class ReActGraph:
    def __init__(self):
        self.workflow = None
        self.state = MyAgentState
        self.tools = tools
        self.init_thinking_node = init_thinking_agent
        self.re_plan_node = replan_agent
        self.do_task_node = do_task
        self.tool_node = tool_node
        self.should_continue_tools = should_end

    def create_graph(self):
        # Create the state graph
        logger.info("___Creating workflow graph___")

        workflow = StateGraph(self.state)

        # Add nodes
        workflow.add_node(INIT_THINKING_NODE, self.init_thinking_node)
        workflow.add_node(RE_PLAIN_NODE, self.re_plan_node)
        workflow.add_node(DO_TASK_NODE, self.do_task_node)
        workflow.add_node(TOOL_NODE, self.tool_node)

        # Set entry point
        workflow.set_entry_point(INIT_THINKING_NODE)

        # Add edges
        workflow.add_edge(INIT_THINKING_NODE, DO_TASK_NODE)
        workflow.add_edge(RE_PLAIN_NODE, DO_TASK_NODE)
        workflow.add_conditional_edges(DO_TASK_NODE, self.should_continue_tools,
                                       {RE_PLAIN_NODE: RE_PLAIN_NODE, TOOL_NODE: TOOL_NODE, END: END})
        workflow.add_edge(TOOL_NODE, DO_TASK_NODE)

        self.workflow = workflow.compile()

        logger.info("___Finished creating workflow graph___")
        return self.workflow

    def print_mermaid(self):
        # Generate and log the Mermaid diagram
        try:
            logger.info("___Printing mermaid graph___")
            mermaid_diagram = self.workflow.get_graph().draw_mermaid()
            logger.info("Mermaid diagram:")
            logger.info(mermaid_diagram)

            logger.info("___Saving mermaid graph to file___")
            current_dir = Path(__file__).parent.absolute()
            project_root = current_dir.parent.parent
            mermaid_dir_path = os.path.join(project_root, "mermaid")
            mermaid_path = os.path.join(mermaid_dir_path, "react_mermaid.mmd")

            ## Save the diagram to a file
            with open(mermaid_path, "w") as f:
                f.write(mermaid_diagram)
                f.close()

            logger.info("___Finished printing mermaid graph___")

        except Exception as e:
            print(f"Error generating Mermaid diagram: {str(e)}")

    async def chat(self, init: str):
        initial_state = {"messages": [HumanMessage(content=init)], "thinking_steps": [], "needs_replanning": False,
                         "previous_thinking": []}

        if self.workflow is None:
            self.create_graph()
            self.print_mermaid()

        try:
            # Run the graph
            result = await self.workflow.ainvoke(initial_state)

            logger.info(f"--- AGENT: Execution completed ---")
            if result.get("messages"):
                logger.info(f"--- AGENT: Final message: {result.get('messages', [])[-1]} ---")

            return {"status": "completed", "messages": result.get("messages", [])}

        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return {"status": "error", "error": str(e), "messages": initial_state["messages"]}
