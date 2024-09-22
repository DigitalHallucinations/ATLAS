# ATLAS/Tools/ToolManager.py

import asyncio
import json
import inspect
import importlib.util
import sys
from datetime import datetime
from ATLAS.config import ConfigManager
from modules.Tools.tool_event_system import event_system

config_manager = ConfigManager()
logger = config_manager.logger 

def get_required_args(function):
    logger.info("getting required args")
    sig = inspect.signature(function)
    return [param.name for param in sig.parameters.values()
            if param.default == param.empty and param.name != 'self']

def load_function_map_from_current_persona(current_persona):
    persona_name = current_persona["name"]
    maps_path = f'modules/Personas/{persona_name}/Toolbox/maps.py'
    module_name = f'persona_{persona_name}_maps'

    spec = importlib.util.spec_from_file_location(module_name, maps_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module.function_map

def load_functions_from_json(current_persona):
    persona_name = current_persona["name"]
    functions_json_path = f'modules/Personas/{persona_name}/Toolbox/functions.json'

    try:
        with open(functions_json_path, 'r') as file:
            functions = json.load(file)
            logger.info("Functions loaded from file")
            return functions
    except FileNotFoundError:
        logger.info(f"Functions JSON file not found for persona: {persona_name}")
    except json.JSONDecodeError as e:
        logger.info(f"Error decoding JSON from {functions_json_path}: {e}")

    return {}

async def use_tool(user, conversation_id, message, conversation_history, function_map, functions, current_persona, temperature_var, top_p_var, conversation_manager, config_manager):
    logger.info(f"use_tool called for user: {user}, conversation_id: {conversation_id}")
    logger.debug(f"Full message: {message}")
    
    if message.get("function_call"):
        logger.info(f"Function call detected: {message['function_call']['name']}")
        function_response, error_occurred = await handle_function_call(user, conversation_id, message, conversation_history, function_map)

        logger.info(f"Function call response: {function_response}")
        logger.info(f"Error occurred: {error_occurred}")

        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        conversation_history.add_response(user, conversation_id, function_response, current_time)
        logger.info("Function call response added to responses table.")

        formatted_function_response = f"System Message: The function call was executed successfully with the following results: {message['function_call']['name']}: {function_response} If needed, you can make another tool call for further processing or multi-step requests. Provide the answer to the user's question, a summary or ask for further details."

        logger.debug(f"Formatted function response: {formatted_function_response}")

        messages = conversation_history.get_history(user, conversation_id)
        logger.debug(f"Conversation history: {messages}")

        new_text = await call_model_with_new_prompt(formatted_function_response, current_persona, messages, temperature_var, top_p_var, functions, config_manager)
        
        logger.info(f"Model response: {new_text}")

        if new_text is None:
            logger.warning("Model returned None response")
            new_text = "Tool Manager says: Sorry, I couldn't generate a meaningful response. Please try again or provide more context."

        logger.info("Assistant: %s", new_text)

        if new_text:
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            conversation_history.add_message(user, conversation_id, "assistant", new_text, current_time)
            logger.info("Assistant message added to conversation history.")

        return new_text
    return None

async def handle_function_call(user, conversation_id, message, conversation_history, function_map):
    logger.info(f"handle_function_call for user: {user}, conversation_id: {conversation_id}")
    logger.debug(f"Full message: {message}")
    
    function_name = message["function_call"]["name"]
    function_args_json = message["function_call"].get("arguments", "{}")

    try:
        function_args = json.loads(function_args_json)
        logger.info(f"Function name: {function_name}")
        logger.debug(f"Function args: {function_args}")
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON: {e}")
        return f"Error: Invalid JSON in function arguments: {e}", True

    if function_name in function_map:
        required_args = get_required_args(function_map[function_name])
        logger.info(f"Required args for {function_name}: {required_args}")
        logger.info(f"Provided args: {list(function_args.keys())}")
        missing_args = set(required_args) - set(function_args.keys())

        if missing_args:
            logger.error(f"Not all required arguments provided for {function_name}. Missing: {', '.join(missing_args)}")
            return f"Error: Not all required arguments provided for {function_name}. Missing: {', '.join(missing_args)}", True

        try:
            logger.info(f"Calling function {function_name} with arguments {function_args}")
            if asyncio.iscoroutinefunction(function_map[function_name]):
                function_response = await function_map[function_name](**function_args)
            else:
                function_response = function_map[function_name](**function_args)
            logger.info(f"Function response: {function_response}")
            
            # Publish event for code execution
            if function_name == "execute_python":
                event_system.publish("code_executed", function_args['command'], function_response)
                logger.info("Published code_executed event")
            
            return function_response, False
        except Exception as e:
            logger.error(f"Exception during function call {function_name}: {e}", exc_info=True)
            return f"Error: Exception during function call {function_name}: {e}", True

    logger.error(f"Function {function_name} not found in function map.")
    return None, True

async def call_model_with_new_prompt(prompt, current_persona, messages, temperature_var, top_p_var, functions, config_manager):
    logger.info("call_model_with_new_prompt called")
    logger.info(f"Prompt: {prompt}")
    logger.debug(f"Messages: {messages}")
    
    response = await config_manager.provider_manager.generate_response(
        messages=messages + [{"role": "user", "content": prompt}],
        model=config_manager.provider_manager.get_current_model(),
        temperature=temperature_var,
        top_p=top_p_var,
        functions=functions
    )
    
    logger.debug(f"Response: {response}")

    if response:
        return response
    else:
        logger.error(f"Failed to get valid response from model.")
        return None