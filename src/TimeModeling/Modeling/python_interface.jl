export devito_interface

########################################### Logging utility ########################################################
_op_str(fw::Bool) = fw ? "F" : "F'"

########################################## Python pycall wrappers with lock #########################################

_outtype(::Nothing, ::Integer, type) = type

function _outtype(b::Bool, n::Integer, type)
    T = b ? PyArray : PyObject
    IT = n==1 ? (T,) : (T, T)
    print("IT: ")
    print(IT)
    print("\n")
    return Tuple{type, IT...}
end

function _outtype_isoelastic(b::Bool, n::Integer, type)

    if n == 2
        return Tuple{type, type, type}
    else 
        return Tuple{type, type, type, type}
    end
end

function wrapcall_data_isoelastic(func, dim, args...;kw...) 

    rtype = _outtype_isoelastic(get(kw, :illum, nothing), dim, PyArray)
    out = rlock_pycall(func, rtype, args...;kw...)

    # tup = isa(out, Tuple)
    # The returned array `out` is a Python Row-Major array with dimension (time, rec).
    # Unlike standard array we want to keep this ordering in julia (time first) so we need to
    # make a wrapper around the pointer, to flip the dimension the re-permute the dimensions.

    shot1 = out[1]
    shot1 = PermutedDimsArray(unsafe_wrap(Array, shot1.data, reverse(size(shot1))), length(size(shot1)):-1:1)
    shot2 = out[2]
    shot2 = PermutedDimsArray(unsafe_wrap(Array, shot2.data, reverse(size(shot2))), length(size(shot2)):-1:1)
    shot3 = out[3]
    shot3 = PermutedDimsArray(unsafe_wrap(Array, shot3.data, reverse(size(shot3))), length(size(shot3)):-1:1)

    if dim == 3
        shot4 = out[4]
        shot4 = PermutedDimsArray(unsafe_wrap(Array, shot4.data, reverse(size(shot4))), length(size(shot4)):-1:1)
        out = (shot1, shot2, shot3, shot4)
    else
        out = (shot1, shot2, shot3)
    end

    return out
end

function wrapcall_data(func, args...;kw...)
    rtype = _outtype(get(kw, :illum, nothing), 1, PyArray)
    out = rlock_pycall(func, rtype, args...;kw...)

    tup = isa(out, Tuple)
    # The returned array `out` is a Python Row-Major array with dimension (time, rec).
    # Unlike standard array we want to keep this ordering in julia (time first) so we need to
    # make a wrapper around the pointer, to flip the dimension the re-permute the dimensions.
    shot = tup ? out[1] : out
    shot = PermutedDimsArray(unsafe_wrap(Array, shot.data, reverse(size(shot))), length(size(shot)):-1:1)
    # Check what to return
    out = tup ? (shot, out[2]) : shot
    # print("\nPrint dentro wrap depois shot")
    # print("\nout: ")
    # print(typeof(out))
    # print("\nout[1]: ")
    # print(typeof(out[1]))

    return out
end

function wrapcall_weights(func, args...;kw...)
    rtype = _outtype(get(kw, :illum, nothing), 1, PyArray)
    out = rlock_pycall(func, rtype, args...;kw...)
    return out
end

function wrapcall_wf(func, args...;kw...)
    rtype = _outtype(get(kw, :illum, nothing), 1, Array{Float32})
    out = rlock_pycall(func, rtype, args...;kw...)
    return out
end

function wrapcall_grad(func, args...;kw...)
    rtype = _outtype(get(kw, :illum, nothing), 2, PyArray)
    out = rlock_pycall(func, rtype, args...;kw...)
    return out
end

# legacy
wrapcall_function = wrapcall_grad

# d_obs = Pr*F*Ps'*q
function devito_interface(modelPy::PyObject, srcGeometry::Geometry, srcData::Array, recGeometry::Geometry, recData::Nothing, dm::Nothing, options::JUDIOptions, illum::Bool, fw::Bool; recvGeometry::Union{Geometry, Nothing}=nothing)
    judilog("Pr*$(_op_str(fw))*Ps'*q")
    # Interpolate input data to computational grid
    dtComp = convert(Float32, modelPy."critical_dt")
    qIn = time_resample(srcData, srcGeometry, dtComp)

    # Set up coordinates with devito dimensions
    src_coords = setup_grid(srcGeometry, modelPy.shape)
    rec_coords = setup_grid(recGeometry, modelPy.shape)

    if options.mc
        rec_coords = (rec_coords, setup_grid(recvGeometry, modelPy.shape))
    end

    # Devito call
    if modelPy.is_elastic
        print("Elastic Execution\n")
        return wrapcall_data_isoelastic(ac."forward_rec", modelPy.dim, modelPy, src_coords, qIn, rec_coords, fw=fw, space_order=options.space_order, f0=options.f0, illum=illum, mc=options.mc, par=options.par)
    end
    # print("e também saiu\n")
    return wrapcall_data(ac."forward_rec", modelPy, src_coords, qIn, rec_coords, fw=fw, space_order=options.space_order, f0=options.f0, illum=illum, par=options.par)
end

# u = F*Ps'*q
function devito_interface(modelPy::PyObject, srcGeometry::Geometry, srcData::Array, recGeometry::Nothing, recData::Nothing, dm::Nothing, options::JUDIOptions, illum::Bool, fw::Bool)
    judilog("$(_op_str(fw))*Ps'*q")
    # Interpolate input data to computational grid
    dtComp = convert(Float32, modelPy."critical_dt")
    qIn = time_resample(srcData, srcGeometry, dtComp)

    # Set up coordinates with devito dimensions
    src_coords = setup_grid(srcGeometry, modelPy.shape)

    # Devito call
    return wrapcall_wf(ac."forward_no_rec", modelPy, src_coords, qIn, fw=fw, space_order=options.space_order, illum=illum, par=options.par)
end

# d_obs = Pr*F*u
function devito_interface(modelPy::PyObject, srcGeometry::Nothing, srcData::Array, recGeometry::Geometry, recData::Nothing, dm::Nothing, options::JUDIOptions, illum::Bool, fw::Bool)
    judilog("Pr*$(_op_str(fw))*u")
    # Interpolate input data to computational grid
    dtComp = convert(Float32, modelPy."critical_dt")

    # Set up coordinates with devito dimensions
    rec_coords = setup_grid(recGeometry, modelPy.shape)

    # Devito call
    return wrapcall_data(ac."forward_wf_src", modelPy, srcData, rec_coords, fw=fw, space_order=options.space_order, f0=options.f0, illum=illum, par=options.par)
end

# u_out = F*u_in
function devito_interface(modelPy::PyObject, srcGeometry::Nothing, srcData::Array, recGeometry::Nothing, recData::Nothing, dm::Nothing, options::JUDIOptions, illum::Bool, fw::Bool)
    judilog("$(_op_str(fw))*u_in")
    # Interpolate input data to computational grid
    dtComp = convert(Float32, modelPy."critical_dt")

    # Devito call
    return wrapcall_wf(ac."forward_wf_src_norec", modelPy, srcData, fw=fw, space_order=options.space_order, illum=illum, par=options.par)
end

# d_lin = J*dm
function devito_interface(modelPy::PyObject, srcGeometry::Geometry, srcData::Array, recGeometry::Geometry,
                          recData::Nothing, dm::Union{PhysicalParameter, Array}, options::JUDIOptions, illum::Bool, fw::Bool)
    judilog("J($(_op_str(fw)), q)*dm")
    # Interpolate input data to computational grid
    dtComp = convert(Float32, modelPy."critical_dt")
    qIn = time_resample(srcData, srcGeometry, dtComp)

    # Set up coordinates with devito dimensions
    #origin = get_origin(modelPy)
    src_coords = setup_grid(srcGeometry, modelPy.shape)
    rec_coords = setup_grid(recGeometry, modelPy.shape)

    # Devito call
    return wrapcall_data(ac."born_rec", modelPy, src_coords, qIn, rec_coords, fw=fw,
                  space_order=options.space_order, ic=options.IC, f0=options.f0, illum=illum)
end

# dm = J'*d_lin
function devito_interface(modelPy::PyObject, srcGeometry::Geometry, srcData::Array, recGeometry::Geometry,
                          recData::Array, dm::Nothing, options::JUDIOptions, illum::Bool, fw::Bool)
    judilog("J($(_op_str(fw)), q)'*d_lin")
    # Interpolate input data to computational grid
    dtComp = convert(Float32, modelPy."critical_dt")
    qIn = time_resample(srcData, srcGeometry, dtComp)
    dIn = time_resample(recData, recGeometry, dtComp)

    # Set up coordinates with devito dimensions
    src_coords = setup_grid(srcGeometry, modelPy.shape)
    rec_coords = setup_grid(recGeometry, modelPy.shape)
    length(options.frequencies) == 0 ? freqs = nothing : freqs = options.frequencies
    return wrapcall_grad(ac."J_adjoint", modelPy,
                  src_coords, qIn, rec_coords, dIn, fw=fw, t_sub=options.subsampling_factor,
                  space_order=options.space_order, checkpointing=options.optimal_checkpointing,
                  freq_list=freqs, ic=options.IC, is_residual=true,
                  dft_sub=options.dft_subsampling_factor[1], f0=options.f0, illum=illum)
end

######################################################################################################################################################

# d_obs = Pr*F*Pw'*w - modeling w/ extended source
function devito_interface(modelPy::PyObject, weights::Array, srcData::Array, recGeometry::Geometry, recData::Nothing,
                          dm::Nothing, options::JUDIOptions, illum::Bool, fw::Bool)
    judilog("Pr*$(_op_str(fw))*Pw'*w")
    weights = pad_array(reshape(weights, modelPy.shape), modelPy.padsizes; mode=:zeros)
    # Interpolate input data to computational grid
    dtComp = convert(Float32, modelPy."critical_dt")
    qIn = time_resample(srcData, recGeometry, dtComp)

    # Set up coordinates with devito dimensions
    rec_coords = setup_grid(recGeometry, modelPy.shape)

    # Devito call
    return wrapcall_data(ac."forward_rec_w", modelPy, weights, qIn, rec_coords,
                         fw=fw, space_order=options.space_order, f0=options.f0, illum=illum)
end

# dw = Pw*F'*Pr'*d_obs - adjoint modeling w/ extended source
function devito_interface(modelPy::PyObject, recGeometry::Geometry, recData::Array, srcData::Array, ::Nothing, ::Nothing, options::JUDIOptions, illum::Bool, fw::Bool)
    judilog("Pw*$(_op_str(fw))*Pr'*d_obs")
    # Interpolate input data to computational grid
    dtComp = convert(Float32, modelPy."critical_dt")
    dIn = time_resample(recData, recGeometry, dtComp)
    qIn = time_resample(srcData, recGeometry, dtComp)

    # Set up coordinates with devito dimensions
    rec_coords = setup_grid(recGeometry, modelPy.shape)

    # Devito call
    return wrapcall_weights(ac."adjoint_w", modelPy, rec_coords, dIn, qIn,
                            fw=fw, space_order=options.space_order, f0=options.f0, illum=illum)
end

# Jacobian of extended source modeling: d_lin = J*dm
function devito_interface(modelPy::PyObject, weights::Array, srcData::Array, recGeometry::Geometry, recData::Nothing,
                          dm::Union{PhysicalParameter, Array}, options::JUDIOptions, illum::Bool, fw::Bool)
    judilog("Jw($(_op_str(fw)), q)*dm")
    weights = pad_array(reshape(weights, modelPy.shape), modelPy.padsizes; mode=:zeros)
    # Interpolate input data to computational grid
    dtComp = convert(Float32, modelPy."critical_dt")
    qIn = time_resample(srcData, recGeometry, dtComp)

    # Set up coordinates with devito dimensions
    rec_coords = setup_grid(recGeometry, modelPy.shape)

    # Devito call
    return wrapcall_data(ac."born_rec_w", modelPy, weights, qIn, rec_coords,
                         fw=fw, space_order=options.space_order, ic=options.IC, f0=options.f0, illum=illum)
end

# Adjoint Jacobian of extended source modeling: dm = J'*d_lin
function devito_interface(modelPy::PyObject, weights::Array, srcData::Array, recGeometry::Geometry, recData::Array, dm::Nothing, options::JUDIOptions, illum::Bool, fw::Bool)
    judilog("Jw($(_op_str(fw)), q)'*d_lin")
    weights = pad_array(reshape(weights, modelPy.shape), modelPy.padsizes; mode=:zeros)
    # Interpolate input data to computational grid
    dtComp = convert(Float32, modelPy."critical_dt")
    qIn = time_resample(srcData, recGeometry, dtComp)
    dIn = time_resample(recData, recGeometry, dtComp)

    # Set up coordinates with devito dimensions
    rec_coords = setup_grid(recGeometry, modelPy.shape)
    length(options.frequencies) == 0 ? freqs = nothing : freqs = options.frequencies
    return wrapcall_grad(ac."J_adjoint", modelPy,
                  nothing, qIn, rec_coords, dIn, fw=fw, t_sub=options.subsampling_factor,
                  space_order=options.space_order, checkpointing=options.optimal_checkpointing,
                  freq_list=freqs, ic=options.IC, ws=weights, is_residual=true,
                  dft_sub=options.dft_subsampling_factor[1], f0=options.f0, illum=illum)
end
