import time

def fill_history_data(history_data, action):
    # Ensure history_data is a list
    if not isinstance(history_data, list):
        history_data = []
    
    # If action is None, return the history_data as is
    if action is None:
        return history_data
    
    # If action is a string, prepend the timestamp and action to the history_data
    if isinstance(action, str):
        current_time = time.strftime("%I:%M %p")  # Format: "HH:MM AM/PM"
        return [f"{current_time} - {action}"] + history_data
    
    # Handle unsupported types for action
    raise ValueError("Action must be a string or None")

    