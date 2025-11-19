import os
import json
import time
import traceback
from openai import OpenAI
import httpx
from datetime import datetime
from typing import Callable, List, Optional
from langchain_core.tools import BaseTool
from seea.utils.base import StateManager
from seea.utils.common import CYAN, RESET, encode_image
from seea.agents.agent.base import BaseChatAgent
from seea.utils.logger import get_logger
logger = get_logger(__name__)


class ChatAgent(BaseChatAgent):
    def __init__(self,
                 name: str = "chatagent",
                 system_prompt: str = "You are a helpful assistant.",
                 model: Optional[str] = None,
                 api_key: Optional[str] = None,
                 api_base: Optional[str] = None,
                 image_size: Optional[List[int]] = None,
                 max_size: int = 1024*768,
                 tool_choice: Optional[str] = "auto",
                 tools: Optional[List[Callable]] = None,
                 function_to_schema: Optional[Callable] = None,
                 parallel_tool_calls: bool = False,
                 max_tokens: int = 1024,
                 temperature: float = 0.35,
                 top_p: float = 0.9,
                 repetition_penalty: float = 1.05,
                 stop: Optional[List[str]] = None,
                 keep_message_history: bool = True,
                 max_steps: int = 50,
                 save_folder: str = "assets/logs",
                 stream=False,
                 debug_mode: bool = False,
                 use_tool_role: bool = True
                 ):
        super().__init__(name=name)
        self.system_prompt = system_prompt
        if self.system_prompt:
            self._initialize_messages()
        logger.info(CYAN + f"{self.name}: system_prompt={self.system_prompt}" + RESET)
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.image_size = image_size
        self.max_size = max_size
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.max_tokens = max_tokens
        self.stop = stop
        self.keep_message_history = keep_message_history
        self.save_folder = save_folder
        self.stream = stream
        self.debug_mode = debug_mode
        self.state_manager = StateManager()
        if self.save_folder and not os.path.exists(save_folder):
            os.makedirs(save_folder, exist_ok=True)
        
        # Explicitly create an httpx client, disabling environment proxy usage
        http_client = httpx.Client(trust_env=False)
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base,
            http_client=http_client # Pass the explicit client
        )
        # Tools
        self.tool_choice = tool_choice
        self.tools_list = []
        self.executable_functions_list = {}
        self.parallel_tool_calls = parallel_tool_calls
        self.function_to_schema = function_to_schema
        self.max_steps = max_steps
        self.num_steps = 0
        # Model compatibility settings
        self.use_tool_role = use_tool_role
        self._check_model_compatibility()
        if tools and function_to_schema:
            self._initialize_tools(tools)

    def _check_model_compatibility(self):
        """Automatically set compatibility options based on the model.
        
        Checks if the current model supports messages with the 'tool' role and configures automatically based on the detection result.
        OpenAI series models (e.g., GPT-4o) use the 'tool' role by default,
        while most open-source models (e.g., Llama, Gemma, Mistral, etc.) use the 'user' role.
        """
        # List of models that do not support the 'tool' role; tool information needs to be converted to the 'user' role.
        tool_role_unsupported = [
            "llama", "gemma", "qwen", "mistral", "yi", "glm", "baichuan"
        ]
        
        if self.model and not self.use_tool_role:
            return
            
        # Check if the model name contains an identifier for a model that does not support the 'tool' role.
        if self.model:
            model_lower = self.model.lower()
            for unsupported in tool_role_unsupported:
                if unsupported in model_lower:
                    self.use_tool_role = False
                    logger.info(f"Model {self.model} does not support the 'tool' role, will use 'user' role instead.")
                    break
        
        # OpenAI series models support the 'tool' role by default.
        if self.model and any(x in self.model.lower() for x in ["gpt", "o1"]):
            self.use_tool_role = True
            
        if self.debug_mode:
            logger.info(f"Model {self.model} configuration: use_tool_role={self.use_tool_role}")

    def initialize(self):
        self.num_steps = 0
        self._initialize_messages()

    def _initialize_messages(self):
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.complete_messages = [{"role": "system", "content": self.system_prompt}]
        self.current_image_path = None

    def _initialize_tools(self, tools: List[Callable]):
        if self.function_to_schema:
            for func in tools:
                self.add_tool(func, self.function_to_schema(func))
        else:
            raise ValueError("function_to_schema is not provided.")

    def add_tool(self, func: Callable, schema):
        self.tools_list.append(schema)
        func_name = schema.get("function").get("name")
        self.executable_functions_list[func_name] = func
    
    def set_system_prompt(self, system_prompt):
        self.system_prompt = system_prompt
        self._initialize_messages()
        logger.info(CYAN + f"{self.name}: system_prompt={self.system_prompt}" + RESET)
    
    def get_system_prompt(self):
        return self.system_prompt

    def get_tools_list(self):
        return self.tools_list

    def post_process(self, content):
        # Remove stop_token and subsequent content from content
        if self.stop == None:
            return content
        for stop_token in self.stop:
            observation_index = content.find(stop_token)
            if observation_index != -1:
                content = content[:observation_index]
                logger.info(CYAN + f"{self.name}: post-processed content={content}" + RESET)
        return content
    
    def _sanitize_messages(self, messages: List[dict]):
        """Ensures the message sequence conforms to the OpenAI / vllm Chat API specification.
        Mainly prevents illegal combinations where a 'tool' message directly follows an 'assistant' message
        that did not previously contain 'tool_calls'.
        If such an illegal 'tool' message is encountered, its role is converted to 'user',
        and it is sent as text to avoid a 400 error.
        """
        if not self.use_tool_role or self.model not in ["gpt-4o", "o1", "o1-mini"]:
            # If not using 'tool' role, or the model does not require strict checking, return original messages directly.
            return messages
            
        sanitized: List[dict] = []
        allow_tool: bool = False  # Allow 'tool' message only if the previous 'assistant' message contained 'tool_calls'.
        for msg in messages:
            role = msg.get("role")
            if role == "assistant":
                allow_tool = bool(msg.get("tool_calls"))
                sanitized.append(msg)
            elif role == "tool":
                if allow_tool:
                    # Legal 'tool' message
                    sanitized.append(msg)
                else:
                    # Downgrade illegal 'tool' message to 'user' plain text.
                    downgraded_msg = {
                        "role": "user",
                        "content": msg.get("content", "")
                    }
                    sanitized.append(downgraded_msg)
                # After a 'tool' message, consecutive 'tool' messages are no longer allowed until a new 'assistant' with 'tool_calls'.
                allow_tool = False
            else:
                # 'user', 'system', etc. roles
                allow_tool = False
                sanitized.append(msg)
        return sanitized

    def run(self, messages):
        try:
            if not self.keep_message_history:
                self._initialize_messages()
            # Check if there is a system message in messages.
            has_system_message = any(message.get("role") == "system" for message in messages)
            self.complete_messages.extend(messages)
            processed_messages = []
            if not has_system_message:
                processed_messages = [{"role": "system", "content": self.system_prompt}]
            # convert image to base64 if exists
            for message in messages:
                if (message.get("role") == "user" or message.get("role") == "tool"):
                    if isinstance(message.get("content"), list):
                        for a in message["content"]:
                            if isinstance(a, dict) and a.get("type") == "image_url":
                                image_path = a.get("image_url")
                                base64_image = self.encode_image(image_path)
                                a["image_url"] = {"url": f"data:image/jpeg;base64,{base64_image}"}
                    elif isinstance(message.get("content"), dict) and "image" in message.get("content"):
                        image_path_list = message.get("content").get("image")
                        if isinstance(image_path_list, str):
                            image_path_list = [image_path_list]
                        text = message.get("content").get("text")
                        message = {
                            "role": message.get("role"),
                            "content": [{"type": "text", "text": text}]}
                        for image_path in image_path_list:
                            base64_image = self.encode_image(image_path)
                            message["content"].append({"type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})
                processed_messages.append(message)
            # Perform a validity check on the messages to be sent, to prevent illegal 'tool' messages.
            processed_messages = self._sanitize_messages(processed_messages)
            # logger.info(f"processed_message:{processed_messages}")
            start_time = time.time()
            response = self.client.chat.completions.create(
                model=self.model,
                messages=processed_messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                extra_body={"repetition_penalty": self.repetition_penalty},
                stop=self.stop or None,
                stream=self.stream  # Add stream parameter
            )
            cost = (time.time() - start_time) * 1000
            content = response.choices[0].message.content
            self.complete_messages.append({"role": "assistant", "content": content})
            logger.info(CYAN + f"{self.name}: content={content}, cost={cost:.1f}ms" + RESET)
            return self.post_process(content)
        except Exception as e:
            logger.error(f"{self.name}: {traceback.format_exc()}")
            return f"An Error {e} occurred. Plese retry."

    def batch_run(self, messages, n: int = 1, logprobs: bool = False):
        try:
            # Check if there is a system message in messages.
            has_system_message = any(message.get("role") == "system" for message in messages)
            processed_messages = []
            if not has_system_message:
                processed_messages = [{"role": "system", "content": self.system_prompt}]
            # convert image to base64 if exists
            for message in messages:
                if (message.get("role") == "user" or message.get("role") == "tool"):
                    if isinstance(message.get("content"), list):
                        for a in message["content"]:
                            if isinstance(a, dict) and a.get("type") == "image_url":
                                image_path = a.get("image_url")
                                base64_image = self.encode_image(image_path)
                                a["image_url"] = {"url": f"data:image/jpeg;base64,{base64_image}"}
                    elif isinstance(message.get("content"), dict) and "image" in message.get("content"):
                        image_path_list = message.get("content").get("image")
                        if isinstance(image_path_list, str):
                            image_path_list = [image_path_list]
                        text = message.get("content").get("text")
                        message = {
                            "role": message.get("role"),
                            "content": [{"type": "text", "text": text}]}
                        for image_path in image_path_list:
                            base64_image = self.encode_image(image_path)
                            message["content"].append({"type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})
                processed_messages.append(message)
            start_time = time.time()
            response = self.client.chat.completions.create(
                model=self.model,
                messages=processed_messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                extra_body={"repetition_penalty": self.repetition_penalty},
                stop=self.stop or None,
                stream=self.stream,  # Add stream parameter
                n=n,
                logprobs=logprobs
            )
            # Construct return structure, including each generated text and logprobs
            result = []
            for choice in response.choices:
                generated_text = self.post_process(choice.message.content)
                # TODO: Process logprobs
                logprobs_list = [token_logprob.logprob for token_logprob in choice.logprobs.content] if logprobs else []
                result.append({
                    "response": generated_text,
                    "logprobs": logprobs_list
                })
            cost = (time.time() - start_time) * 1000
            logger.info(CYAN + f"{self.name}: result={result}, cost={cost:.1f}ms" + RESET)
            return result
        except Exception as e:
            logger.error(f"{self.name}: {traceback.format_exc()}")
            return f"An Error {e} occurred. Plese retry."
        
    
    def reflection_run(self, messages, force_prefix_think: bool, reflection_prefix: str) -> Optional[str]:
        try:
            has_system_message = any(message.get("role") == "system" for message in messages)
            processed_messages = []

            if not has_system_message:
                processed_messages.append({"role": "system", "content": self.system_prompt})

            # Process images
            for message in messages:
                if message.get("role") in {"user", "tool"}:
                    if isinstance(message.get("content"), list):
                        for a in message["content"]:
                            if isinstance(a, dict) and a.get("type") == "image_url":
                                image_path = a.get("image_url")
                                base64_image = self.encode_image(image_path)
                                a["image_url"] = {"url": f"data:image/jpeg;base64,{base64_image}"}
                    elif isinstance(message.get("content"), dict) and "image" in message["content"]:
                        image_path_list = message["content"]["image"]
                        if isinstance(image_path_list, str):
                            image_path_list = [image_path_list]
                        text = message["content"].get("text", "")
                        message = {
                            "role": message["role"],
                            "content": [{"type": "text", "text": text}]
                        }
                        for image_path in image_path_list:
                            base64_image = self.encode_image(image_path)
                            message["content"].append({"type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})
                processed_messages.append(message)

            # Inference request
            start_time = time.time()
            response = self.client.chat.completions.create(
                model=self.model,
                messages=processed_messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                extra_body={
                    'force_prefix_think': force_prefix_think,
                    'reflection_prefix': reflection_prefix,
                    'repetition_penalty': self.repetition_penalty
                },
                stop=self.stop or None,
                stream=self.stream
            )
            content = response.choices[0].message.content
            cost = (time.time() - start_time) * 1000
            logger.info(CYAN + f"{self.name}: content={content}, cost={cost:.1f}ms" + RESET)
            return self.post_process(content)
        except Exception as e:
            logger.error(f"{self.name}: {traceback.format_exc()}")
            return None

    
    def process(self, text, image_path_list: Optional[list] = None):
        # Process user input message
        message = {"role": "user"}
        
        # Process multimodal content
        if image_path_list:
            # Use list format to store multimodal content, suitable for OpenAI format
            content_list = [{"type": "text", "text": text}]
            for image_path in image_path_list:
                base64_image = self.encode_image(image_path)
                content_list.append({"type": "image_url", 
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})
            message["content"] = content_list
            
            # Save a simplified version of multimodal content for complete message history
            complete_message = {
                "role": "user",
                "content": {"text": text, "image": image_path_list}
            }
        else:
            # Plain text content
            message["content"] = text
            complete_message = {"role": "user", "content": text}
        
        # Add to message history
        self.messages.append(message)
        self.complete_messages.append(complete_message)
        
        return message
    
    def encode_image(self, image_path):
        self.current_image_path = os.path.basename(image_path)
        # Ensure image path is absolute
        image_path =  os.path.abspath(image_path) if not os.path.isabs(image_path) else image_path
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        self.state_manager.set("env_image_path", image_path)
        base64_image = encode_image(image_path, new_size=self.image_size, max_size=self.max_size)
        return base64_image

    def __call__(self, text, image_path_list=None):
        if isinstance(image_path_list, str):
            image_path_list = [image_path_list]
        return self.chat(text, image_path_list)
    
    def chat(self, text, image_path_list=None):
        try:
            # Handle message history.
            if not self.keep_message_history:
                self._initialize_messages()
            start_time = time.time()
            user_message = self.process(text, image_path_list)
            cost = (time.time() - start_time) * 1000
            user_message = {"role": "user", "content": {"text": text, "image_path_list": image_path_list}}
            logger.info(CYAN + f"{self.name}: user_message={user_message}, cost={cost:.1f}ms" + RESET)
            return self.infer()
        except:
            logger.error(f"{self.name}: {traceback.format_exc()}")
            return {"role": "assistant", "content": ""}

    def infer(self): 
        try:
            start_time = time.time()
            self.num_steps += 1
            if self.num_steps > self.max_steps:
                logger.warning(f"Maximum number of calls reached ({self.max_steps}), stopping further calls.")
                return {"role": "assistant", "content": f"Maximum number of calls reached ({self.max_steps})"}
            if self.tool_choice == 'auto' and len(self.tools_list) > 0:
                # Recheck the validity of self.messages before calling the API
                sanitized_messages = self._sanitize_messages(self.messages)
                if self.debug_mode:
                    logger.info(f"Messages sent to model: {sanitized_messages}")
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=sanitized_messages,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                    extra_body={"repetition_penalty": self.repetition_penalty},
                    tool_choice="auto",
                    tools=self.tools_list,
                    parallel_tool_calls=self.parallel_tool_calls,
                    stop=self.stop or None,
                    stream=self.stream
                    )
            else:
                # Recheck the validity of self.messages before calling the API
                sanitized_messages = self._sanitize_messages(self.messages)
                if self.debug_mode:
                    logger.info(f"Messages sent to model: {sanitized_messages}")
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=sanitized_messages,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                    extra_body={"repetition_penalty": self.repetition_penalty},
                    stop=self.stop or None,
                    stream=self.stream
                    )
            if self.stream and self.parser:
                content = ""
                tool_calls = []
                tool_messages = []
                for parsed in self.parser(response):
                    if parsed["type"] == "answer":
                        self.reply(parsed["data"])
                    elif parsed["type"] == "tool_call":
                        logger.info("Tool call: %s", parsed["data"])
                        try:
                            tool_call = parsed["data"]
                            tool_function_name = tool_call.get("name", "")
                            arguments = tool_call.get("arguments", {})
                            # Add a unique ID for each tool call
                            tool_calls.append({
                                "function": {"name": tool_function_name, "arguments": arguments}, 
                                "id": f"call_{time.time()}_{len(tool_calls)}"
                            })
                        except Exception as e:
                            logger.error(f"{self.name}: Failed to parse tool call: {str(e)}")
                            logger.error(f"{self.name}: {traceback.format_exc()}")
                            tool_message = self._create_tool_message(
                                f"{parsed['data']} failed to parse as JSON, please check the format.")
                            tool_messages.append(tool_message)
                    elif parsed["type"] == "complete_response":
                        content = self.post_process(parsed["data"])
                cost = (time.time() - start_time) * 1000
                message = {"role": "assistant", "content": content}
                logger.info(CYAN + f"{self.name}: message={message}, cost={cost:.1f}ms" + RESET)
                self.messages.append(message)
                self.complete_messages.append(message)
                if tool_messages:
                    logger.info(CYAN + f"{self.name}: tool_messages={tool_messages}" + RESET)
                    self.messages.extend(tool_messages)
                    self.complete_messages.extend(tool_messages)
                if tool_calls:
                    message = self.handle_tool_calls(tool_calls)
            else:
                cost = (time.time() - start_time) * 1000
                message = response.choices[0].message.to_dict()
                logger.info(CYAN + f"{self.name}: message={message}, cost={cost:.1f}ms" + RESET)
                message["content"] = self.post_process(message["content"])
                tool_calls = message.get("tool_calls", [])
                if "tool_calls" in message:
                    del message["tool_calls"]
                self.messages.append(message)
                self.complete_messages.append(message)
                if self.tool_choice == 'auto' and tool_calls is not None and len(tool_calls) > 0:
                    message = self.handle_tool_calls(tool_calls)
                elif self.parser:
                    parsed_tool_calls = self.parse_tool_call(message.get("content", ""))
                    if parsed_tool_calls:
                        # Convert parsed tool calls to a unified format
                        formatted_tool_calls = []
                        for function_name, arguments in parsed_tool_calls:
                            formatted_tool_calls.append({
                                "function": {"name": function_name, "arguments": arguments},
                                "id": f"call_{time.time()}_{len(formatted_tool_calls)}"
                            })
                        message = self.handle_tool_calls(formatted_tool_calls)
            if self.save_folder:
                self.save_data()

            return message
        except Exception as e:
            logger.error(f"{self.name}: Error in infer: {str(e)}")
            logger.error(f"{self.name}: {traceback.format_exc()}")
            return {"role": "assistant", "content": f"An error occurred: {str(e)}"}
    
    def save_data(self):
        try:
            # Get current date and time, formatted as YYYY-MM-DD HH:MM
            current_datetime = datetime.now().strftime("%Y-%m-%d %H-%M")
            # Create a filename with the date suffix
            filename = f"{self.name}_{self.model}_{current_datetime}.jsonl"
            # Use os.path.join() to combine the save folder path and filename into a complete save path
            save_path = os.path.join(str(self.save_folder), str(filename))
            
            # Use with statement to ensure the file is closed correctly, avoiding resource leaks
            with open(save_path, "a", encoding="utf-8") as f:
                json.dump(self.complete_messages, f, ensure_ascii=False, indent=4)
                f.write('\n')
                # Flush data to disk immediately
                f.flush()
                
            logger.debug(f"{self.name}: Data saved to {save_path}")
        except Exception as e:
            logger.error(f"{self.name}: Error saving data: {str(e)}")

    def _get_role_for_tool_message(self):
        """Return the appropriate role based on model compatibility.
        
        Returns:
            str: "tool" or "user" based on model compatibility.
        """
        return "tool" if self.use_tool_role else "user"
        
    def _create_tool_message(self, content, name=None):
        """Create a tool message, selecting the appropriate role based on model compatibility.
        
        Args:
            content: Message content.
            name: Tool name, used only when use_tool_role is True.
            
        Returns:
            dict: Formatted message dictionary.
        """
        message = {"role": self._get_role_for_tool_message(), "content": content}
        if name and self.use_tool_role:
            message["name"] = name
        return message

    def close(self):
        """
        Closes the underlying OpenAI client and releases resources.
        """
        if hasattr(self, 'client') and self.client is not None:
            logger.info(f"Closing OpenAI client for agent {self.name}...")
            try:
                # The OpenAI client (based on httpx) has a close method.
                self.client.close()
                logger.info(f"OpenAI client for agent {self.name} closed successfully.")
            except Exception as e:
                logger.error(f"Error closing OpenAI client for agent {self.name}: {traceback.format_exc()}")
            finally:
                self.client = None # Optional: clear the client reference after closing
        else:
            logger.info(f"No active OpenAI client to close for agent {self.name}.")


class FunctionCallAgent(ChatAgent):
    def __init__(self,
                 recursive: bool = True,
                 max_actions: int = 30,
                 text_world: bool = False,
                 parser: Optional[Callable] = None,
                 observation_format: str = "Observation: {}",
                 fixed_perception: bool = False,
                 *args, **kwargs
                 ):
        """
        Args:q
            recursive: Whether to call tools recursively. If True, continuously call tools until no more tool calls are made; otherwise, call tools only once.
            max_steps: Maximum number of steps.
            text_world: Whether it is a text world. If True, automatically complete tool call results; otherwise, use actual tool call results.
            parser: Parser.
            fixed_perception: Whether to fix perception. If True, call the perception tool to get the latest environment information after each non-perception tool call.
        """
        super().__init__(*args, **kwargs)
        self.recursive = recursive
        self.max_actions = max_actions
        self.num_actions = 0
        self.text_world = text_world
        self.tool_call_id = None
        self.reply_func = None
        self.reply_func_name = None
        self.parser = parser
        self.observation_format = observation_format
        self.fixed_perception = fixed_perception

    def initialize(self):
        self.num_steps = 0
        self.num_actions = 0
        self._initialize_messages()

    def get_tool_call_id(self):
        return self.tool_call_id

    def add_reply_function(self, func, name):
        # Add a function to reply to the user
        self.reply_func = func
        self.reply_func_name = name

    def reply(self, response):
        # Call the reply tool function to reply to the user
        try:
            if self.reply_func and self.reply_func_name:
                arguments = {"message": response, "wait": False}
                if isinstance(self.reply_func, BaseTool):
                    result = self.reply_func.invoke(input=arguments)
                else:
                    result = self.reply_func(**arguments)
                tool_message = self._create_tool_message(result, self.reply_func_name)
                return tool_message
        except Exception as e:
            logger.error(f"{self.name}: Error in reply: {str(e)}")
            logger.error(f"{self.name}: {traceback.format_exc()}")
        return None

    def handle_tool_calls(self, tool_calls):
        try:                
            # Initialize a unified tool message
            combined_tool_message = self._create_tool_message("")
            
            # Process all tool calls
            has_perception = False
            combined_results = []
            combined_images = []
            
            for tool_call in tool_calls:
                # Get tool call ID (if any)
                tool_call_id = tool_call.get("id", None)
                self.tool_call_id = tool_call_id  # Save the last tool call ID
                
                # Get function information
                function_info = tool_call.get("function", {})
                tool_function_name = function_info.get("name")
                arguments = function_info.get("arguments", {})
                
                # If arguments is a string, try to parse it as JSON
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse arguments: {arguments}")
                        arguments = {}
                
                # Execute tool call but do not add to message history
                result = self._execute_function_call(tool_function_name, arguments, tool_call_id)
                
                # Add result to combined results list
                if isinstance(result, dict) and "image" in result:
                    # If result contains an image
                    image_path = result.get("image")
                    if isinstance(image_path, list):
                        combined_images.extend(image_path)
                    else:
                        combined_images.append(image_path)
                    result = result.get("text", "")
                combined_results.append(self.observation_format.format(result))
                
                # Check if there is a perception tool
                if "perception" in tool_function_name:
                    has_perception = True
            
            # If fixed perception is needed, execute perception but do not return immediately
            if self.fixed_perception and not has_perception:
                perception_result = self._execute_perception()
                if perception_result:
                    if isinstance(perception_result, dict) and "image" in perception_result:
                        # If result contains an image
                        image_path = perception_result.get("image")
                        if image_path:
                            combined_images.append(image_path)
                        perception_result = perception_result.get("text", "")
                    combined_results.append(self.observation_format.format(f"[perception]: {perception_result}"))
            
            # Combine all results into one message
            if combined_images:
                # Save complete message record, using unified format
                complete_tool_message = {
                    "role": "user",
                    "content": {
                        "text": "\n".join(combined_results),
                        "image": combined_images
                    }
                }
                self.complete_messages.append(complete_tool_message)
                
                try:
                    # Encode images for model message
                    model_content = [{"type": "text", "text": "\n".join(combined_results)}]
                    
                    # Add all images
                    for image_path in combined_images:
                        base64_image = self.encode_image(image_path)
                        model_content.append({"type": "image_url", 
                                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})
                    
                    model_tool_message = {"role": "user", "content": model_content}
                    self.messages.append(model_tool_message)
                    combined_tool_message = model_tool_message  # Update the returned message to the encoded message
                except Exception as e:
                    # If image encoding fails, add only the text part
                    logger.error(f"Image encoding failed: {str(e)}")
                    text_only_message = self._create_tool_message("\n".join(combined_results))
                    self.messages.append(text_only_message)
                    self.complete_messages[-1]["content"] = "\n".join(combined_results)
                    combined_tool_message = text_only_message
            else:
                # Plain text result
                combined_tool_message["content"] = "\n".join(combined_results)
                self.messages.append(combined_tool_message)
                self.complete_messages.append(combined_tool_message)
        except Exception as e:
            logger.error(f"{self.name}: Error in handle_tool_calls: {str(e)}")
            logger.error(f"{self.name}: {traceback.format_exc()}")
            combined_tool_message = self._create_tool_message(
                self.observation_format.format(f"Error processing tool call: {str(e)}"))
            self.messages.append(combined_tool_message)
            self.complete_messages.append(combined_tool_message)
        finally:
            if self.debug_mode:
                logger.info(CYAN + f"{self.name}: tool_message={combined_tool_message}" + RESET)
            # Recursively continue the conversation
            if not self.state_manager.get("eval_completed", False) and self.num_steps < self.max_steps and self.num_actions < self.max_actions and self.recursive:  
                return self.infer()
            return combined_tool_message
    
    def _execute_function_call(self, name, arguments, tool_call_id=None):
        """Execute tool function without adding to message history"""     
        # Increment step count
        self.num_actions += 1
        logger.info(f"Current tool call count: {self.num_actions}/{self.max_actions}")

        """Execute tool function without adding to message history"""
        # Check if maximum step limit is exceeded
        if self.num_actions > self.max_actions:
            logger.warning(f"Maximum tool call limit reached ({self.max_actions}), stopping further tool calls.")
            return "Maximum tool call limit reached"
            
        # Check if the function exists in the tool dictionary
        if name in self.executable_functions_list:
            tool_function = self.executable_functions_list[name]
            # Execute tool function or simulate execution
            if self.text_world:
                # In text world mode, return simulated tool execution result
                result = f"[Simulated tool execution] {name}({json.dumps(arguments, ensure_ascii=False)})"
            else:
                # Actually execute the tool function
                if isinstance(tool_function, BaseTool):
                    result = tool_function.invoke(input=arguments)
                else:
                    result = tool_function(**arguments)
        elif not name:
            result = f"An error occurred while retrieving the tool name({name}) with arguments({arguments}). "
        else:
            result = f"Error: function {name} not registered with arguments({arguments})."
        
        logger.info(f"Tool execution result: {result}")
        return result
    
    def _execute_perception(self):
        """Execute perception function without adding to message history"""
        try:
            # Get environment image path from global state
            if self.state_manager.get("env_image_path", None):
                image_path = self.state_manager.get("env_image_path")
                predictions = None
            else:
                return None
            
            # Check if environment image path has changed, skip perception if no change
            if self.current_image_path and self.current_image_path == os.path.basename(str(image_path)):
                return None
            
            # Update current image path
            self.current_image_path = os.path.basename(str(image_path))
            
            # Construct perception result
            text = "Here is the latest image from the environment."
            if predictions and predictions != "[]":
                text += f" Prediction results in the image is: {predictions}"
            
            logger.info(CYAN + f"{self.name}: perception text={text}, image_path={image_path}" + RESET)
            
            return {"text": text, "image": image_path}
        except Exception as e:
            logger.error(f"{self.name}: Error in perception: {e}")
            logger.error(f"{self.name}: {traceback.format_exc()}")
            return f"Error in perception: {str(e)}"
    
    def perception(self):
        """Execute perception and return result"""
        perception_result = self._execute_perception()
        
        if not perception_result:
            # If no new perception result, continue conversation
            if not self.state_manager.get("eval_completed", False) and self.recursive and self.num_steps < self.max_steps:
                return self.infer()
            else:
                return self.messages[-1] if self.messages else {"role": "assistant", "content": "Unable to obtain perception information"}
        
        # Construct perception message
        if isinstance(perception_result, dict) and "image" in perception_result:
            text = perception_result.get("text", "")
            image_path = perception_result.get("image")
            
            # Create tool message
            complete_tool_message = {
                "role": "user",
                "content": {"text": text, "image": image_path}
            }
            
            # Add image path to complete message history
            self.complete_messages.append(complete_tool_message)
            
            try:
                # Encode image for model message
                model_content = [{"type": "text", "text": text}]
                
                # Add image
                base64_image = self.encode_image(image_path)
                model_content.append({"type": "image_url", 
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})
                
                model_tool_message = {"role": "user", "content": model_content}
                self.messages.append(model_tool_message)
                tool_message = model_tool_message
            except Exception as e:
                # If image encoding fails, add only text part
                logger.error(f"Image encoding failed: {str(e)}")
                text_only_message = self._create_tool_message(text)
                self.messages.append(text_only_message)
                self.complete_messages[-1]["content"] = text
                tool_message = text_only_message
        else:
            # Plain text result
            tool_message = self._create_tool_message(perception_result)
            self.messages.append(tool_message)
            self.complete_messages.append(tool_message)
        
        # Recursively continue conversation
        if not self.state_manager.get("eval_completed", False) and self.recursive and self.num_steps < self.max_steps and self.num_actions < self.max_actions:
            return self.infer()
        return tool_message

    def parse_tool_call(self, content):
        tool_calls = []
        try:
            if not self.parser:
                return tool_calls
                
            result = self.parser(content)
            if isinstance(result, dict) and "action" in result:
                for action in result.get("action", []):
                    function_name = action.get("name", "")
                    arguments = action.get("arguments", {})
                    if function_name:
                        tool_calls.append((function_name, arguments))
            return tool_calls
        except Exception as e:
            logger.error(f"{self.name}: Failed to parse tool call: {str(e)}")
            logger.error(f"{self.name}: {traceback.format_exc()}")
            tool_message = self._create_tool_message(f"Failed to parse tool call: {str(e)}")
            self.messages.append(tool_message)
            self.complete_messages.append(tool_message)
            return tool_calls
