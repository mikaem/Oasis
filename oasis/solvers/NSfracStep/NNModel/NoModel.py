__author__ = 'Anna Haley <ahaley@mie.utoronto.ca>'
__date__ = '2020-05-04'
__copyright__ = 'Copyright (C) 2020 ' + __author__
__license__ = 'GNU Lesser GPL version 3 or any later version'

from dolfin import Constant

__all__ = ['nn_setup', 'nn_update']

def nn_setup(**NS_namespace):
    return dict(nunn_=Constant(0))


def nn_update(**NS_namespace):
    pass
