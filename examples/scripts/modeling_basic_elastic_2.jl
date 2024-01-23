# Example for basic 3D modeling:
# The receiver positions and the source wavelets are the same for each of the four experiments.
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January 2017
#

using JUDI, SegyIO, LinearAlgebra, Serialization


n = (200, 120)    # (x,y,z) or (x,z)
d = (10., 10.)
o = (0., 0.)
nlayers = 3
nbl = 50
space_order = 8

vs = zeros(Float32,n)
rho = zeros(Float32,n)
b = zeros(Float32,n)

# Define a velocity profile. The velocity is in km/s
vp_top = 1.5
vp_bottom = 3.5
# define a velocity profile in km/s
v = zeros(Float32,n)
v[:,1:40] .= 1.5f0
v[:,41:80] .= 2.5f0
v[:,81:120] .= 3.5f0

vs .= 0.5 .* v
vs[:,1:40] .= 0.0f0

rho .= 0.31f0.*(v.*1000.0f0).^0.25f0


m = (1f0 ./ v).^2
model = Model(n, d, o, m; rho=rho, vs=vs, nb=nbl)

domain_size = 1990f0
nsrc = 1

nxrec = n[1]
xrec_p = range(10f0, stop=domain_size, length=nxrec)
yrec_p = 0f0
zrec_p = range(700f0, 400f0, length=nxrec)

xrec_v = range(0f0, stop=domain_size, length=nxrec)
yrec_v = 0f0
zrec_v = range(400f0, 400f0, length=nxrec)

# receiver sampling and recording time
timeR = 1500f0   # receiver recording time [ms]
dtR = 1.5f0    # receiver sampling interval

par="lam-mu"
opt = Options(mc=false, par=par)

if opt.mc
    # Set up receiver structure
    recpGeometry = Geometry(xrec_p, yrec_p, zrec_p; dt=dtR, t=timeR, nsrc=nsrc)
    recvGeometry = Geometry(xrec_v, yrec_v, zrec_v; dt=dtR, t=timeR, nsrc=nsrc)

    recGeometry = Geometry(recpGeometry, recvGeometry)
else
    recGeometry = Geometry(xrec_p, yrec_p, zrec_p; dt=dtR, t=timeR, nsrc=nsrc)
end
# print("oi")

# # Set up source geometry (cell array with source locations for each shot)
xsrc = 600f0
ysrc = 0f0
zsrc = 0f0

# source sampling and number of time steps
timeS = 1500f0   # source length in [ms]
dtS = 1.5f0     # source sampling interval

# Set up source structure
srcGeometry = Geometry(xsrc, ysrc, zsrc; dt=dtS, t=timeS)

f0 = 0.020f0
wavelet = ricker_wavelet(timeS, dtS, f0)

# ###################################################################################################

# Setup operators'
F = judiModeling(model, srcGeometry, recGeometry; options=opt) # tenho que ver e entender essa formação do operador de modelagem
# F = judiModeling(model) # tenho que ver e entender essa formação do operador de modelagem

q = judiVector(srcGeometry, wavelet)
# Nonlinear modeling

dobs = F*q


# project_name = "stiffness/"
# dobs_dir = "/scratch/projeto-lde/dados/judi-modelling/" 

# println("rec_tau")
# path = dobs_dir * "dobs-normal"
# block_out = judiVector_to_SeisBlock(dobs, q; source_depth_key="SourceDepth")
# segy_write(path, block_out)
# println("fim rec_tau")

# # println("rec_vx")
# # path = dobs_dir * "rec_vx_judi"
# # block_out = judiVector_to_SeisBlock(dobs[2], q; source_depth_key="SourceDepth")
# # segy_write(path, block_out)
# # println("fim rec_vx")

# # println("rec_vz")
# # path = dobs_dir * "rec_vz_judi"
# # block_out = judiVector_to_SeisBlock(dobs[3], q; source_depth_key="SourceDepth")
# # segy_write(path, block_out)
# # println("fim rec_vz")

# println("FINALIZADO")