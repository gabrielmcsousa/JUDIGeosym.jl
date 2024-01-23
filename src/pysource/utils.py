try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable
import numpy as np
from sympy import sqrt

from devito import configuration
from devito.tools import as_tuple

from devito.types.tensor import (TensorFunction, TensorTimeFunction,
                                 VectorFunction, VectorTimeFunction, tens_func)

import numpy as np
from sympy import symbols, Matrix, ones


# Weighting
def weight_fun(w_fun, model, src_coords):
    """
    Symbolic weighting function

    Parameters
    ----------
    w_fun: Tuple(String, Float)
        Weighting fucntion and weight.
    model: Model
        Model structure
    src_coords: Array
        Source coordinates.
    """
    if w_fun is None:
        return 1
    else:
        return weight_srcfocus(model, src_coords, delta=w_fun[1],
                               full=(w_fun[0] == "srcfocus"))


def weight_srcfocus(model, src_coords, delta=.01, full=True):
    """
    Source focusing weighting function
    w(x) = sqrt((||x-xsrc||^2+delta^2)/delta^2)

    Parameters
    ----------
    model: Model
        Model structure
    src_coords: Array
        Source coordinates
    delta: Float
        Reference distance for weights
    """
    w_dim = as_tuple(model.grid.dimensions if full else model.grid.dimensions[-1])
    isrc = tuple(np.float32(model.padsizes[i][0]) + src_coords[0, i] / model.spacing[i]
                 for i in range(model.dim))
    h = np.prod(model.spacing)**(1/model.dim)
    radius = sum((d - isrc[i])**2 for i, d in enumerate(w_dim))
    return sqrt(radius + (delta / h)**2) / (delta/h)


def compute_optalpha(norm_r, norm_Fty, epsilon, comp_alpha=True):
    """
    Compute optimal alpha for WRI

    Parameters
    ----------
    norm_r: Float
        Norm of residual
    norm_Fty: Float
        Norm of adjoint wavefield squared
    epsilon: Float
        Noise level
    comp_alpha: Bool
        Whether to compute the optimal alpha or just return 1
    """
    if comp_alpha:
        if norm_r > epsilon and norm_Fty > 0:
            return norm_r * (norm_r - epsilon) / norm_Fty
        else:
            return 0
    else:
        return 1


def opt_op(model):
    """
    Setup the compiler options for the operator. Dependeing on the devito
    version more or less options can be used for performance, mostly impacting TTI.

    Parameters
    ----------
    model: Model
        Model structure to know if we are in a TTI model
    """
    if configuration['platform'].name in ['nvidiaX', 'amdgpuX']:
        opts = {'openmp': True if configuration['language'] == 'openmp' else None,
                'mpi': configuration['mpi']}
        mode = 'advanced'
    else:
        opts = {'openmp': True, 'par-collapse-ncores': 2, 'mpi': configuration['mpi']}
        mode = 'advanced'
    return (mode, opts)


def nfreq(freq_list):
    """
    Check number of on-the-fly DFT frequencies.
    """
    return 0 if freq_list is None else np.shape(freq_list)[0]


def fields_kwargs(*args):
    """
    Creates a dictionary of {f.name: f} for any field argument that is not None
    """
    kw = {}
    for field in args:
        if field is not None:
            # In some case could be a tuple of fields, such as dft modes
            if isinstance(field, Iterable):
                kw.update(fields_kwargs(*field))
            else:
                try:
                    kw.update({f.name: f for f in field.flat()})
                    continue
                except AttributeError:
                    kw.update({field.name: field})

    return kw


DEVICE = {"id": -1}  # noqa


def set_device_ids(devid):
    DEVICE["id"] = devid


def base_kwargs(dt):
    """
    Most basic keyword arguments needed by the operator.
    """
    if configuration['platform'].name == 'nvidiaX':
        return {'dt': dt, 'deviceid': DEVICE["id"]}
    else:
        return {'dt': dt}


class C_Matrix():

    C_matrix_dependency = {'lam-mu': 'C_lambda_mu', 'vp-vs-rho': 'C_vp_vs_rho',
                           'Ip-Is-rho': 'C_Ip_Is_rho'}

    def __new__(cls, model, parameters):
        c_m_gen = cls.C_matrix_gen(parameters)
        return c_m_gen(model)

    @classmethod
    def C_matrix_gen(cls, parameters):
        return getattr(cls, cls.C_matrix_dependency[parameters])

    def _matrix_init(dim):
        def cij(i, j):
            ii, jj = min(i, j), max(i, j)
            if (ii == jj or (ii <= dim and jj <= dim)):
                return symbols('C%s%s' % (ii, jj))
            return 0

        d = dim*2 + dim-2
        Cij = [[cij(i, j) for i in range(1, d)] for j in range(1, d)]
        return Matrix(Cij)

    @classmethod
    def C_lambda_mu(cls, model):
        def subs3D():
            return {'C11': lmbda + (2*mu),
                    'C22': lmbda + (2*mu),
                    'C33': lmbda + (2*mu),
                    'C44': mu,
                    'C55': mu,
                    'C66': mu,
                    'C12': lmbda,
                    'C13': lmbda,
                    'C23': lmbda}

        def subs2D():
            return {'C11': lmbda + (2*mu),
                    'C22': lmbda + (2*mu),
                    'C33': mu,
                    'C12': lmbda}

        matriz = C_Matrix._matrix_init(model.dim)
        lmbda = model.lam
        mu = model.mu

        subs = subs3D() if model.dim == 3 else subs2D()
        M = matriz.subs(subs)

        M.dlam = cls._generate_Dlam(model)
        M.dmu = cls._generate_Dmu(model)
        M.inv = cls._inverse_C_lam(model)
        return M

    @staticmethod
    def _inverse_C_lam(model):
        def subs3D():
            return {'C11': (lmbda + mu)/(3*lmbda*mu + 2*mu*mu),
                    'C22': (lmbda + mu)/(3*lmbda*mu + 2*mu*mu),
                    'C33': (lmbda + mu)/(3*lmbda*mu + 2*mu*mu),
                    'C44': 1/mu,
                    'C55': 1/mu,
                    'C66': 1/mu,
                    'C12': -lmbda/(6*lmbda*mu + 4*mu*mu),
                    'C13': -lmbda/(6*lmbda*mu + 4*mu*mu),
                    'C23': -lmbda/(6*lmbda*mu + 4*mu*mu)}

        def subs2D():
            return {'C11': (lmbda + mu)/(3*lmbda*mu + 2*mu*mu),
                    'C22': (lmbda + mu)/(3*lmbda*mu + 2*mu*mu),
                    'C33': 1/mu,
                    'C12': -lmbda/(6*lmbda*mu + 4*mu*mu)}

        matrix = C_Matrix._matrix_init(model.dim)
        lmbda = model.lam
        mu = model.mu

        subs = subs3D() if model.dim == 3 else subs2D()
        return matrix.subs(subs)

    @staticmethod
    def _generate_Dlam(model):
        def d_lam(i, j):
            ii, jj = min(i, j), max(i, j)
            if (ii <= model.dim and jj <= model.dim):
                return 1
            return 0

        d = model.dim*2 + model.dim-2
        Dlam = [[d_lam(i, j) for i in range(1, d)] for j in range(1, d)]
        return Matrix(Dlam)

    @staticmethod
    def _generate_Dmu(model):
        def d_mu(i, j):
            ii, jj = min(i, j), max(i, j)
            if (ii == jj):
                if ii <= model.dim:
                    return 2
                else:
                    return 1
            return 0

        d = model.dim*2 + model.dim-2
        Dmu = [[d_mu(i, j) for i in range(1, d)] for j in range(1, d)]
        return Matrix(Dmu)

    @classmethod
    def C_vp_vs_rho(cls, model):
        def subs3D():
            return {'C11': rho*vp*vp,
                    'C22': rho*vp*vp,
                    'C33': rho*vp*vp,
                    'C44': rho*vs*vs,
                    'C55': rho*vs*vs,
                    'C66': rho*vs*vs,
                    'C12': rho*vp*vp - 2*rho*vs*vs,
                    'C13': rho*vp*vp - 2*rho*vs*vs,
                    'C23': rho*vp*vp - 2*rho*vs*vs}

        def subs2D():
            return {'C11': rho*vp*vp,
                    'C22': rho*vp*vp,
                    'C33': rho*vs*vs,
                    'C12': rho*vp*vp - 2*rho*vs*vs}

        matrix = C_Matrix._matrix_init(model.dim)
        vp = model.vp
        vs = model.vs
        rho = 1/model.irho

        subs = subs3D() if model.dim == 3 else subs2D()
        M = matrix.subs(subs)

        M.dvp = cls._generate_Dvp(model)
        M.dvs = cls._generate_Dvs(model)
        M.drho = cls._generate_Drho(model)
        M.inv = cls._inverse_C_vp_vs(model)
        return M

    @staticmethod
    def _inverse_C_vp_vs(model):
        def subs3D():
            return {'C11': (vp*vp - vs*vs)/((rho*vs*vs)*(3*vp*vp - 4*vs*vs)),
                    'C22': (vp*vp - vs*vs)/((rho*vs*vs)*(3*vp*vp - 4*vs*vs)),
                    'C33': (vp*vp - vs*vs)/((rho*vs*vs)*(3*vp*vp - 4*vs*vs)),
                    'C44': 1/(rho*vs*vs),
                    'C55': 1/(rho*vs*vs),
                    'C66': 1/(rho*vs*vs),
                    'C12': (vp*vp - vs*vs)/((rho*vs*vs)*(6*vp*vp - 8*vs*vs)),
                    'C13': (vp*vp - vs*vs)/((rho*vs*vs)*(6*vp*vp - 8*vs*vs)),
                    'C23': (vp*vp - vs*vs)/((rho*vs*vs)*(6*vp*vp - 8*vs*vs))}

        def subs2D():
            return {'C11': (vp*vp - vs*vs)/((rho*vs*vs)*(3*vp*vp - 4*vs*vs)),
                    'C22': (vp*vp - vs*vs)/((rho*vs*vs)*(3*vp*vp - 4*vs*vs)),
                    'C33': 1/(rho*vs*vs),
                    'C12': (vp*vp - vs*vs)/((rho*vs*vs)*(6*vp*vp - 8*vs*vs))}

        matrix = C_Matrix._matrix_init(model.dim)
        vp = model.vp
        vs = model.vs
        rho = 1/model.irho

        subs = subs3D() if model.dim == 3 else subs2D()
        return matrix.subs(subs)

    @staticmethod
    def _generate_Dvp(model):
        def d_vp(i, j):
            ii, jj = min(i, j), max(i, j)
            if (ii <= model.dim and jj <= model.dim):
                return 2*(1/model.irho)*model.vp
            return 0

        d = model.dim*2 + model.dim-2
        Dvp = [[d_vp(i, j) for i in range(1, d)] for j in range(1, d)]
        return Matrix(Dvp)

    @staticmethod
    def _generate_Dvs(model):
        def subs3D():
            return {'C11': 0,
                    'C22': 0,
                    'C33': 0,
                    'C44': 2*rho*vs,
                    'C55': 2*rho*vs,
                    'C66': 2*rho*vs,
                    'C12': -4*rho*vs,
                    'C13': -4*rho*vs,
                    'C23': -4*rho*vs}

        def subs2D():
            return {'C11': 0,
                    'C22': 0,
                    'C33': 2*rho*vs,
                    'C12': -4*rho*vs}

        Dvs = C_Matrix._matrix_init(model.dim)
        rho = 1/model.irho
        vs = model.vs

        subs = subs3D() if model.dim == 3 else subs2D()
        return Dvs.subs(subs)

    @staticmethod
    def _generate_Drho(model):
        def subs3D():
            return {'C11': vp*vp,
                    'C22': vp*vp,
                    'C33': vp*vp,
                    'C44': vs*vs,
                    'C55': vs*vs,
                    'C66': vs*vs,
                    'C12': vp*vp - 2*vs*vs,
                    'C13': vp*vp - 2*vs*vs,
                    'C23': vp*vp - 2*vs*vs}

        def subs2D():
            return {'C11': vp*vp,
                    'C22': vp*vp,
                    'C33': vs*vs,
                    'C12': vp*vp - 2*vs*vs}

        Dvs = C_Matrix._matrix_init(model.dim)
        vp = model.vp
        vs = model.vs

        subs = subs3D() if model.dim == 3 else subs2D()
        return Dvs.subs(subs)

    @classmethod
    def C_Ip_Is_rho(cls, model):
        def subs3D():
            return {'C11': Ip*vp,
                    'C22': Ip*vp,
                    'C33': Ip*vp,
                    'C44': Is*vs,
                    'C55': Is*vs,
                    'C66': Is*vs,
                    'C12': Ip*vp - 2*Is*vs,
                    'C13': Ip*vp - 2*Is*vs,
                    'C23': Ip*vp - 2*Is*vs}

        def subs2D():
            return {'C11': Ip*vp,
                    'C22': Ip*vp,
                    'C33': Is*vs,
                    'C12': Ip*vp - 2*Is*vs}

        matrix = cls._matrix_init(model.dim)
        vp = model.vp
        vs = model.vs
        Ip = model.Ip
        Is = model.Is

        subs = subs3D() if model.dim == 3 else subs2D()
        M = matrix.subs(subs)

        M.dIs = cls._generate_DIs(model)
        M.dIp = cls._generate_DIp(model)

        return M

    @staticmethod
    def _generate_DIp(model):
        def d_Is(i, j):
            ii, jj = min(i, j), max(i, j)
            if (ii <= model.dim and jj <= model.dim):
                return model.vp
            return 0

        d = model.dim*2 + model.dim-2
        D_Is = [[d_Is(i, j) for i in range(1, d)] for j in range(1, d)]
        return Matrix(D_Is)

    @staticmethod
    def _generate_DIs(model):
        def subs3D():
            return {'C11': 0,
                    'C22': 0,
                    'C33': 0,
                    'C44': vs,
                    'C55': vs,
                    'C66': vs,
                    'C12': -2*vs,
                    'C13': -2*vs,
                    'C23': -2*vs}

        def subs2D():
            return {'C11': 0,
                    'C22': 0,
                    'C33': vs,
                    'C12': -2*vs}

        D_Ip = C_Matrix._matrix_init(model.dim)
        vs = model.vs

        subs = subs3D() if model.dim == 3 else subs2D()
        return D_Ip.subs(subs)


def D(self, shift=None):
    """
    Returns the result of matrix D applied over the TensorFunction.
    """
    if not self.is_TensorValued:
        raise TypeError("The object must be a Tensor object")

    M = tensor(self) if self.shape[0] != self.shape[1] else self

    comps = []
    func = tens_func(self)
    for j, d in enumerate(self.space_dimensions):
        comps.append(sum([getattr(M[j, i], 'd%s' % d.name)
                         for i, d in enumerate(self.space_dimensions)]))
    return func._new(comps)


def S(self, shift=None):
    """
    Returns the result of transposed matrix D applied over the VectorFunction.
    """
    if not self.is_VectorValued:
        raise TypeError("The object must be a Vector object")

    derivs = ['d%s' % d.name for d in self.space_dimensions]

    comp = []
    comp.append(getattr(self[0], derivs[0]))
    comp.append(getattr(self[1], derivs[1]))
    if len(self.space_dimensions) == 3:
        comp.append(getattr(self[2], derivs[2]))
        comp.append(getattr(self[1], derivs[2]) + getattr(self[2], derivs[1]))
        comp.append(getattr(self[0], derivs[2]) + getattr(self[2], derivs[0]))
    comp.append(getattr(self[0], derivs[1]) + getattr(self[1], derivs[0]))

    func = tens_func(self)

    return func._new(comp)


def vec(self):
    if not self.is_TensorValued:
        raise TypeError("The object must be a Tensor object")
    if self.shape[0] != self.shape[1]:
        raise Exception("This object is already represented by its vector form.")

    order = ([(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
             if len(self.space_dimensions) == 3 else [(0, 0), (1, 1), (0, 1)])
    comp = [self[o[0], o[1]] for o in order]
    func = tens_func(self)
    return func(comp)


def tensor(self):
    if not self.is_TensorValued:
        raise TypeError("The object must be a Tensor object")
    if self.shape[0] == self.shape[1]:
        raise Exception("This object is already represented by its tensor form.")

    ndim = len(self.space_dimensions)
    M = np.zeros((ndim, ndim), dtype=np.dtype(object))
    M[0, 0] = self[0]
    M[1, 1] = self[1]
    if len(self.space_dimensions) == 3:
        M[2, 2] = self[2]
        M[2, 1] = self[3]
        M[1, 2] = self[3]
        M[2, 0] = self[4]
        M[0, 2] = self[4]
    M[1, 0] = self[-1]
    M[0, 1] = self[-1]

    func = tens_func(self)
    return func._new(M)


def gather(a1, a2):

    expected_a1_types = [int, VectorFunction, VectorTimeFunction]
    expected_a2_types = [int, TensorFunction, TensorTimeFunction]

    if type(a1) not in expected_a1_types:
        raise ValueError("a1 must be a VectorFunction or a Integer")
    if type(a2) not in expected_a2_types:
        raise ValueError("a2 must be a TensorFunction or a Integer")
    if type(a1) is int and type(a2) is int:
        raise ValueError("Both a2 and a1 cannot be Integers simultaneously")

    if type(a1) is int:
        a1_m = Matrix([ones(len(a2.space_dimensions), 1)*a1])
    else:
        a1_m = Matrix(a1)

    if type(a2) is int:
        ndim = len(a1.space_dimensions)
        a2_m = Matrix([ones((3*ndim-3), 1)*a2])
    else:
        a2_m = Matrix(a2)

    if a1_m.cols > 1:
        a1_m = a1_m.T
    if a2_m.cols > 1:
        a2_m = a2_m.T

    return Matrix.vstack(a1_m, a2_m)