from devito.tools import as_tuple

from sources import *


def src_rec(model, u, src_coords=None, rec_coords=None, wavelet=None, nt=None):
    nt = nt or wavelet.shape[0]

    if model.is_elastic:
        namef = as_tuple(u)[1][0].name
    else:
        # namef = as_tuple(u)[1][0].name if model.is_elastic else as_tuple(u)[0].name
        namef = as_tuple(u)[0].name
    src = None
    if src_coords is not None:
        if isinstance(wavelet, PointSource):
            src = wavelet
        else:
            src = PointSource(name="src%s" % namef, grid=model.grid, ntime=nt,
                            coordinates=src_coords)
            src.data[:] = wavelet[:] if wavelet is not None else 0.
    rcv = None
    if rec_coords is not None:
        rcv = Receiver(name="rcv%s" % namef, grid=model.grid, ntime=nt,
                    coordinates=rec_coords)
    return src, rcv


def geom_expr(model, u, src_coords=None, rec_coords=None, wavelet=None, fw=True, nt=None, recv_coords=None):
    """
    Generates the source injection and receiver interpolation.
    This function is fully abstracted and does not care whether this is a
    forward or adjoint wave-equation.
    The source is the source term of the equation
    The receiver is the measurment term

    Therefore, for the adjoint, this function has to be called as:
    src_rec(model, v, src_coords=rec_coords, ...)
    because the data is the sources

    Parameters
    ----------
    model : Model
        Physical model
    u : TimeFunction or tuple
        Wavefield to inject into and read from
    src_coords : Array
        Physical coordinates of the sources
    rec_coords : Array
        Physical coordinates of the receivers
    wavelet: Array
        Data for the source
    fw=True:
        Whether the direction is forward or backward in time
    nt: int
        Number of time steps
    """
    irho = model.irho
    if not model.is_elastic:
        m = model.m * irho
    dt = model.grid.time_dim.spacing
    geom_expr = []
    src, rcv = src_rec(model, u, src_coords, rec_coords, wavelet, nt)
    if src is not None:
        # Elastic inject into diagonal of stress
        if model.is_elastic:
            for ud in as_tuple(u)[1].diagonal():
                geom_expr += src.inject(field=ud.forward, expr=src*dt)
        else:
            # Acoustic inject into pressure
            u_n = as_tuple(u)[0].forward if fw else as_tuple(u)[0].backward
            geom_expr += src.inject(field=u_n, expr=src*dt**2/m)
    # Setup adjoint wavefield sampling at source locations
    if rcv is not None:
        if model.is_elastic:

            if recv_coords is None:
                raise Exception
            
            rec_vx = Receiver(name="rec_vx", grid=model.grid, ntime=nt,
                    coordinates=recv_coords)
            rec_vz = Receiver(name="rec_vz", grid=model.grid, ntime=nt,
                    coordinates=recv_coords)
            if model.grid.dim == 3:
                rec_vy = Receiver(name="rec_vy", grid=model.grid, ntime=nt,
                    coordinates=recv_coords)

            v = u[0]
            rec_term_vx = rec_vx.interpolate(expr=v[0])
            rec_term_vz = rec_vz.interpolate(expr=v[1])
            rec_expr = rec_term_vx + rec_term_vz

            if model.grid.dim == 3:
                rec_term_vy = rec_vy.interpolate(expr=v[2])
                rec_expr += rec_term_vy

            tau_expr = u[1].trace()
            rec_term_tau = rcv.interpolate(expr=tau_expr)
            rec_expr += rec_term_tau
            geom_expr += rec_expr
            # geom_expr +=rec_term_tau

            return geom_expr
        else:
            rec_expr = u[0] if model.is_tti else u
        adj_rcv = rcv.interpolate(expr=rec_expr)
        geom_expr += adj_rcv
    return geom_expr
