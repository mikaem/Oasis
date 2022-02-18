# lots of unused imports, that are all imported and used by the main script
# from .io import *
from oasis.common.io import (
    create_initial_folders,
    save_solution,
    save_tstep_solution_h5,
    save_checkpoint_solution_h5,
    check_if_kill,
    check_if_reset_statistics,
    init_from_restart,
    merge_visualization_files,
    merge_xml_files,
)
from os import makedirs, getcwd, listdir, remove, system, path
from xml.etree import ElementTree as ET
import pickle
import time
import glob
from oasis.problems import info_red

# from .utilities import *
from oasis.common.utilities import (
    A_cache,
    Solver_cache,
    Mat_cache_dict,
    Solver_cache_dict,
    assemble_matrix,
    OasisFunction,
    GradFunction,
    DivFunction,
    CG1Function,
    AssignedVectorFunction,
    LESsource,
    NNsource,
    homogenize,
)
from ufl.tensors import ListTensor
from ufl import Coefficient
import sys
import json


def convert(input):
    if isinstance(input, dict):
        return {convert(key): convert(value) for key, value in input.iter()}
    elif isinstance(input, list):
        return [convert(element) for element in input]
    elif isinstance(input, str):
        return input.encode("utf-8")
    else:
        return input


# Parse command-line keyword arguments
def parse_command_line():
    commandline_kwargs = {}
    for s in sys.argv[1:]:
        if s.count("=") == 1:
            key, value = s.split("=", 1)
        else:
            raise TypeError(
                (
                    s
                    + " Only kwargs separated with '=' sign "
                    + "allowed. See NSdefault_hooks for a range of "
                    + "parameters. Your problem file should contain "
                    + "problem specific parameters."
                )
            )
        try:
            value = json.loads(value)

        except ValueError:
            if value in (
                "True",
                "False",
            ):  # json understands true/false, but not True/False
                value = eval(value)
            elif "True" in value or "False" in value:
                value = eval(value)
        if isinstance(value, dict):
            value = convert(value)

        commandline_kwargs[key] = value
    return commandline_kwargs


# Note to self. To change a dictionary variable through commandline do, e.g.,
# run NSfracStep velocity_update_solver='{"method":"gradient_matrix"}'
