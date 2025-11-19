import abc


class BaseAgent:
    """BaseAgent defines the inference interface `infer` for all types of Agents."""
    def __init__(self, name = None) -> None:
        self.name = self.__class__.__name__ if name is None else name
    
    @abc.abstractmethod    
    def infer(self, *args, **kwargs):
        """Performs inference once.
        
        Returns:
            result_dict (dict): A dict containing the following required fields:
                answer (str): The response text.
        """
        pass


class BaseChatAgent(BaseAgent):
    """BaseChatAgent specifies the inference input parameter `instruction`."""
    
    @abc.abstractmethod    
    def infer(self, instruction, *args, **kwargs):
        pass