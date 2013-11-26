from io import *
import sys, json

# Parse command-line keyword arguments
def parse_command_line():
    commandline_kwargs = {}
    for s in sys.argv[1:]:
        if s.count('=') == 1:
            key, value = s.split('=', 1)
        else:
            raise TypeError(s+" Only kwargs separated with '=' sign allowed. See NSdefault_hooks for a range of parameters. Your problem file should contain problem specific parameters.")
        try:
            value = json.loads(value) 
        except ValueError:
            if value in ("True", "False"): # json understands true/false, but not True/False
                value = eval(value)
        commandline_kwargs[key] = value
    return commandline_kwargs

