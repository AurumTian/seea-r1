# ALFWorld task type prefix mapping
PREFIXES = {
    "pick_and_place": "put",
    "pick_clean_then_place": "clean",
    "pick_heat_then_place": "heat",
    "pick_cool_then_place": "cool",
    "look_at_obj": "examine",
    "pick_two_obj": "puttwo",
}

# Task type template mapping
TASK_TEMPLATES = {
    "pick_and_place": """###Task Description###
1. You need to find and pick up an object, then place it at a specified location.
2. You can only carry one object at a time.
3. If you see an object but cannot interact with it, you may be too far - move closer to its location.
4. Failed actions or actions without any state-changes result in Nothing happens.""",
    
    "pick_clean_then_place": """###Task Description###
1. You need to find an object, clean it, and then place it at a specified location.
2. Cleaning requires using a sink with running water.
3. You can only carry one object at a time.
4. If you see an object but cannot interact with it, you may be too far - move closer to its location.
5. Failed actions or actions without any state-changes result in Nothing happens.""",
    
    "pick_heat_then_place": """###Task Description###
1. You need to find an object, heat it using a microwave, and then place it at a specified location.
2. Heating requires putting the object in a microwave and turning it on.
3. You can only carry one object at a time.
4. If you see an object but cannot interact with it, you may be too far - move closer to its location.
5. Failed actions or actions without any state-changes result in Nothing happens.""",
    
    "pick_cool_then_place": """###Task Description###
1. You need to find an object, cool it using a fridge, and then place it at a specified location.
2. Cooling requires putting the object in a fridge for some time.
3. You can only carry one object at a time.
4. If you see an object but cannot interact with it, you may be too far - move closer to its location.
5. Failed actions or actions without any state-changes result in Nothing happens.""",
    
    "look_at_obj": """###Task Description###
1. You need to find and examine a specific object in the environment.
2. Examining requires looking at the object closely.
3. If you see an object but cannot interact with it, you may be too far - move closer to its location.
4. Failed actions or actions without any state-changes result in Nothing happens.""",
    
    "pick_two_obj": """###Task Description###
1. You need to find and pick up two different objects, then place them at specified locations.
2. You can only carry one object at a time, so you'll need to make multiple trips.
3. If you see an object but cannot interact with it, you may be too far - move closer to its location.
4. Failed actions or actions without any state-changes result in Nothing happens.""",
}


def identify_task_type(task: str) -> str:
    """
    Identify ALFWorld task type based on task description
    
    Args:
        task (str): ALFWorld task description
    
    Returns:
        str: Task type, such as pick_and_place, pick_clean_then_place, etc.
    """
    task_lower = task.lower()
    
    for task_type, prefix in PREFIXES.items():
        if task_type in task_lower:
            return task_type
    # Default return pick_and_place type
    return "pick_and_place"


def get_task_template(task_type: str) -> str:
    """
    Get the template for a specific task type
    
    Args:
        task_type (str): Task type
    
    Returns:
        str: Template for the specific task type
    """
    return TASK_TEMPLATES.get(task_type, TASK_TEMPLATES["pick_and_place"])