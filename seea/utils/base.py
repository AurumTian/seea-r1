import threading

class StateManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.shared_state = {}  # Shared state dictionary
            cls._instance.thread_local = threading.local()  # Thread-local storage
        return cls._instance
    
    def get(self, key, default=None, thread_local=True):
        """Get state value
        
        Args:
            key: Key name
            default: Default value
            thread_local: Whether to get from thread-local storage
        """
        if thread_local:
            if not hasattr(self.thread_local, 'state'):
                self.thread_local.state = {}
            return self.thread_local.state.get(key, default)
        return self.shared_state.get(key, default)
    
    def set(self, key, value, thread_local=True):
        """Set state value
        
        Args:
            key: Key name
            value: Value
            thread_local: Whether to store in thread-local storage
        """
        if thread_local:
            if not hasattr(self.thread_local, 'state'):
                self.thread_local.state = {}
            self.thread_local.state[key] = value
        else:
            self.shared_state[key] = value
        
    def update(self, thread_local=True, **kwargs):
        """Batch update state
        
        Args:
            thread_local: Whether to update to thread-local storage
            **kwargs: Key-value pairs
        """
        if thread_local:
            if not hasattr(self.thread_local, 'state'):
                self.thread_local.state = {}
            self.thread_local.state.update(kwargs)
        else:
            self.shared_state.update(kwargs)
            
    def clear_thread_local(self):
        """Clear the local storage of the current thread"""
        if hasattr(self.thread_local, 'state'):
            self.thread_local.state = {}
            
    def get_thread_local_state(self):
        """Get all local states of the current thread"""
        if not hasattr(self.thread_local, 'state'):
            self.thread_local.state = {}
        return self.thread_local.state
    
    def remove(self, key, thread_local=True):
        """Remove state value
        
        Args:
            key: Key name
            thread_local: Whether to remove from thread-local storage
        """
        if thread_local:
            if hasattr(self.thread_local, 'state') and key in self.thread_local.state:
                del self.thread_local.state[key]
        else:
            if key in self.shared_state:
                del self.shared_state[key]