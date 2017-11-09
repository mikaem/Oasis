import pytest
import subprocess
import re

########
# These tests only check if the untested code runs without breaking, and does not
# test any properties. These should be replace with better tests.
########

@pytest.mark.parametrize("num_p", [1, 4])
@pytest.mark.parametrize("solver", ["IPCS_ABCN", "IPCS_ABE", "Chorin", "BDFPC", "BDFPC_Fast"])
@pytest.mark.parametrize("les_model", ["Smagorinsky", "Wale",
                                       "DynamicLagrangian",
                                       "ScaleDepDynamicLagrangian"])
def test_LES(num_p, solver, les_model):
    cmd = "mpirun -np {} python NSfracStep.py solver={} T=0.0001 dt=0.00005 les_model={};"
    cmd = "cd ..; " + cmd + " cd tests"
    subprocess.check_output(cmd.format(num_p, solver, les_model), shell=True)


@pytest.mark.parametrize("num_p", [1, 4])
@pytest.mark.parametrize("solver", ["IPCS_ABCN", "IPCS_ABE", "Chorin", "BDFPC", "BDFPC_Fast"])
@pytest.mark.parametrize("problem", ["Channel", "Cylinder", "DrivenCavity",
                                     "DrivenCavity3D", "FlowPastSphere3D",
                                     "LaminarChannel", "Lshape", "Skewed2D",
                                     "SkewedFlow", "TaylorGreen2D", "TaylorGreen3D"])
def test_demo_NSfracStep(num_p, solver, problem):
    cmd = "mpirun -np {} python NSfracStep.py solver={} T=0.0001 dt=0.00005 problem={};"
    cmd = "cd ..; " + cmd + " cd tests"
    subprocess.check_output(cmd.format(num_p, solver, problem), shell=True)
