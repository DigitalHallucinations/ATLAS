# ATLAS/Tools/ToolManager.py

import asyncio
import json
import inspect
import importlib.util
import sys
from datetime import datetime
from modules.logging.logger import setup_logger
from modules.Tools.tool_event_system import event_system

from ATLAS.config import ConfigManager
logger = setup_logger(__name__)

_function_map_cache = {}

def get_required_args(function):
    logger.info("Retrieving required arguments for the function.")
    sig = inspect.signature(function)
    return [
        param.name for param in sig.parameters.values()
        if param.default == param.empty and param.name != 'self'
    ]

def load_function_map_from_current_persona(current_persona, *, refresh=False):
    logger.info("Attempting to load function map from current persona.")
    if not current_persona or "name" not in current_persona:
        logger.error("Current persona is None or does not have a 'name' key.")
        return None

    persona_name = current_persona["name"]
    maps_path = f'modules/Personas/{persona_name}/Toolbox/maps.py'
    module_name = f'persona_{persona_name}_maps'

    try:
        if refresh:
            logger.info(
                "Refresh requested for persona '%s'; clearing cached module and function map.",
                persona_name,
            )
            sys.modules.pop(module_name, None)
            _function_map_cache.pop(persona_name, None)

        if not refresh and persona_name in _function_map_cache:
            logger.info(
                "Returning cached function map for persona '%s' without reloading module.",
                persona_name,
            )
            return _function_map_cache[persona_name]

        module = sys.modules.get(module_name)

        if module is None:
            logger.info(
                "Module '%s' not found in sys.modules; loading from '%s'.",
                module_name,
                maps_path,
            )
            spec = importlib.util.spec_from_file_location(module_name, maps_path)
            if spec is None or spec.loader is None:
                raise ImportError(
                    f"Could not load specification for persona '{persona_name}' from {maps_path}"
                )
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        else:
            logger.info(
                "Reusing already loaded module '%s' from sys.modules for persona '%s'.",
                module_name,
                persona_name,
            )

        if hasattr(module, 'function_map'):
            logger.info(
                "Function map successfully loaded for persona '%s': %s",
                persona_name,
                module.function_map,
            )
            _function_map_cache[persona_name] = module.function_map
            return module.function_map
        else:
            logger.warning(
                "No 'function_map' found in maps.py for persona '%s'.",
                persona_name,
            )
            _function_map_cache.pop(persona_name, None)
            return None
    except FileNotFoundError:
        logger.error(f"maps.py file not found for persona '{persona_name}' at path: {maps_path}")
    except Exception as e:
        logger.error(f"Error loading function map for persona '{persona_name}': {e}", exc_info=True)
    return None

def load_functions_from_json(current_persona):
    logger.info("Attempting to load functions from JSON for the current persona.")
    if not current_persona or "name" not in current_persona:
        logger.error("Current persona is None or does not have a 'name' key.")
        return None

    persona_name = current_persona["name"]
    functions_json_path = f'modules/Personas/{persona_name}/Toolbox/functions.json'

    try:
        with open(functions_json_path, 'r') as file:
            functions = json.load(file)
            logger.info(f"Functions successfully loaded from JSON for persona '{persona_name}': {functions}")
            return functions
    except FileNotFoundError:
        logger.error(f"functions.json file not found for persona '{persona_name}' at path: {functions_json_path}")
    except json.JSONDecodeError as e:
        logger.error(f"JSON decoding error in functions.json for persona '{persona_name}': {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error loading functions for persona '{persona_name}': {e}", exc_info=True)
    return None

async def use_tool(
    user,
    conversation_id,
    message,
    conversation_history,
    function_map,
    functions,
    current_persona,
    temperature_var,
    top_p_var,
    frequency_penalty_var,
    presence_penalty_var,
    conversation_manager,
    config_manager=None
):
    logger.info(f"use_tool called for user: {user}, conversation_id: {conversation_id}")
    logger.info(f"Message received: {message}")

    if config_manager is None:
        logger.debug("No ConfigManager supplied to use_tool; instantiating a new one lazily.")
        config_manager = ConfigManager()

    if not isinstance(message, dict):
        normalized_message = {}
        for attr in ("function_call", "tool_calls", "tool_call"):
            value = getattr(message, attr, None)
            if value is not None:
                normalized_message[attr] = value
        message = normalized_message

    def _safe_get(target, key, default=None):
        if isinstance(target, dict):
            return target.get(key, default)
        return getattr(target, key, default)

    if not message.get("function_call"):
        tool_calls_payload = message.get("tool_calls")
        tool_call_entry = None
        if isinstance(tool_calls_payload, list) and tool_calls_payload:
            tool_call_entry = tool_calls_payload[0]
        elif tool_calls_payload:
            tool_call_entry = tool_calls_payload
        if tool_call_entry is None:
            tool_call_entry = message.get("tool_call")

        if tool_call_entry:
            function_payload = _safe_get(tool_call_entry, "function") or tool_call_entry
            name = _safe_get(function_payload, "name") or _safe_get(tool_call_entry, "name")
            arguments = _safe_get(function_payload, "arguments")
            if arguments is None:
                arguments = _safe_get(tool_call_entry, "arguments")
            if name:
                if not isinstance(arguments, str):
                    try:
                        arguments = json.dumps(arguments)
                    except (TypeError, ValueError):
                        arguments = str(arguments)
                message = dict(message)
                message["function_call"] = {"name": name, "arguments": arguments}

    if message.get("function_call"):
        function_name = message["function_call"].get("name")
        logger.info(f"Function call detected: {function_name}")
        function_args_json = message["function_call"].get("arguments", "{}")
        logger.info(f"Function arguments (JSON): {function_args_json}")

        try:
            function_args = json.loads(function_args_json)
            logger.info(f"Function arguments (parsed): {function_args}")
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON for function arguments: {e}", exc_info=True)
            return f"Error: Invalid JSON in function arguments: {e}", True

        if function_name in function_map:
            required_args = get_required_args(function_map[function_name])
            logger.info(f"Required arguments for function '{function_name}': {required_args}")
            provided_args = list(function_args.keys())
            logger.info(f"Provided arguments for function '{function_name}': {provided_args}")
            missing_args = set(required_args) - set(function_args.keys())

            if missing_args:
                logger.error(f"Missing required arguments for function '{function_name}': {missing_args}")
                return (
                    f"Error: Missing required arguments for function '{function_name}': {', '.join(missing_args)}",
                    True
                )

            try:
                logger.info(f"Calling function '{function_name}' with arguments: {function_args}")
                func = function_map[function_name]
                if asyncio.iscoroutinefunction(func):
                    function_response = await func(**function_args)
                else:
                    function_response = func(**function_args)
                logger.info(f"Function '{function_name}' executed successfully. Response: {function_response}")

                # Publish event for specific functions if needed
                if function_name == "execute_python":
                    command = function_args.get('command')
                    event_system.publish("code_executed", command, function_response)
                    logger.info("Published 'code_executed' event.")

                # Add the function response to conversation history
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                conversation_history.add_response(user, conversation_id, function_response, current_time)
                logger.info("Function response added to conversation history.")

                # Prepare the system message for the model
                formatted_function_response = (
                    f"System Message: The function call was executed successfully with the following results: "
                    f"{function_name}: {function_response} "
                    "If needed, you can make another tool call for further processing or multi-step requests. "
                    "Provide the answer to the user's question, a summary, or ask for further details."
                )
                logger.info(f"Formatted function response for model: {formatted_function_response}")

                # Retrieve updated conversation history
                messages = conversation_history.get_history(user, conversation_id)
                logger.info(f"Conversation history: {messages}")

                # Call the model with the new prompt
                new_text = await call_model_with_new_prompt(
                    formatted_function_response,
                    current_persona,
                    messages,
                    temperature_var,
                    top_p_var,
                    frequency_penalty_var,
                    presence_penalty_var,
                    functions,
                    config_manager
                )
                
                logger.info(f"Model response after function execution: {new_text}")

                if new_text is None:
                    logger.warning("Model returned None response. Using default fallback message.")
                    new_text = (
                        "Tool Manager says: Sorry, I couldn't generate a meaningful response. "
                        "Please try again or provide more context."
                    )

                if new_text:
                    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    conversation_history.add_message(user, conversation_id, "assistant", new_text, current_time)
                    logger.info("Assistant's message added to conversation history.")

                return new_text

            except Exception as e:
                logger.error(f"Exception during function '{function_name}' execution: {e}", exc_info=True)
                return f"Error: Exception during function '{function_name}' execution: {e}", True

        else:
            logger.error(f"Function '{function_name}' not found in function map.")
            return f"Error: Function '{function_name}' not found.", True

    return None

async def call_model_with_new_prompt(
    prompt,
    current_persona,
    messages,
    temperature_var,
    top_p_var,
    frequency_penalty_var,
    presence_penalty_var,
    functions,
    config_manager=None
):
    logger.info("Calling model with new prompt after function execution.")
    logger.info(f"Prompt: {prompt}")

    if config_manager is None:
        logger.debug("No ConfigManager supplied to call_model_with_new_prompt; instantiating a new one lazily.")
        config_manager = ConfigManager()
    
    try:
        response = await config_manager.provider_manager.generate_response(
            messages=messages + [{"role": "user", "content": prompt}],
            model=config_manager.provider_manager.get_current_model(),
            temperature=temperature_var,
            top_p=top_p_var,
            frequency_penalty=frequency_penalty_var,
            presence_penalty=presence_penalty_var,
            functions=functions
        )
        logger.info(f"Model's response: {response}")
        return response
    except Exception as e:
        logger.error(f"Error calling model with new prompt: {e}", exc_info=True)
        return None
