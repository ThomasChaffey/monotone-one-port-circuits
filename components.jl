
using LinearAlgebra
using DataFrames
using CSV

include("fixed_point_iterations.jl")

struct UnstableException <: Exception
    var::String
end

Base.showerror(io::IO, e::UnstableException) = print(io, e.var, " iterations" )

## device definitions
function G_diode(v; n = 1, Vt = 25.85e-3, Is = 1e-14)
    i = Is*(exp.(v/n/Vt) .- 1)
end

@inline function R_diode(i, n = 1, Vt = 25.85e-3, Is = 1e-14)
    v = n*Vt*log.(i/Is .+ 1)
end

@inline function dR_diode(i, n = 1, Vt = 25.85e-3, Is = 1e-14)
    dv = n*Vt/(1 .+ i/Is)/Is
end

# forward linear operator of RC circuit
# (conductance form)
S_RC(N) = GC(N) # old notation
function GC(N, R = 1, C = 1)
    # differentiate q_c to give i_c
    diff = Array(Bidiagonal(vec(ones(1, N)), -1*vec(ones(1, N - 1)), :L))
    diff[1, end] = -1
    diff = N*diff # divide by dt
    S = C*diff + (1/R)*I
    return S
end

# inverse linear operator of RC circuit
# (resistance form)
function RC(N, R = 1, C = 1)
    # differentiate q_c to give i_c
    diff = Array(Bidiagonal(vec(ones(1, N)), -1*vec(ones(1, N - 1)), :L))
    diff[1, end] = -1
    diff = N*diff # divide by dt
    S = C*diff + (1/R)*I
    return inv(S)
end

# α-resolvent of RC circuit (resistance form)
# Using this makes it heaps quicker!
function R_RC(N, α, R = 1, C = 1)
    # differentiate q_c to give i_c
    diff = Array(Bidiagonal(vec(ones(1, N)), -vec(ones(1, N - 1)), :L))
    diff[1, end] = -1
    diff = N*diff # divide by dt
    S = (1/R)*I + C*diff 
    return inv(α*I + S)*S
end

# α-resolvent of RC circuit (conductance form)
function S_GC(N, α; R = 1, C = 1)
    # differentiate q_c to give i_c
    diff = Array(Bidiagonal(vec(ones(1, N)), -vec(ones(1, N - 1)), :L))
    diff[1, end] = -1
    diff = N*diff # divide by dt
    S = C*diff + (1/R)*I
    return inv(I + α*S)
end
#
# resolvent of a nonlinear resistor, using the guarded Newton method.  
# Takes a scalar i-v function and its derivative, and an offset vstar. Lower bound is currently
# hard coded for the Shockley equation.
function res_diode(u, R, dR, vstar; α = 0.5, ϵ = 0.0001)
        x = zeros(size(u))
        Threads.@threads for i = 1:length(u)
                x[i] = prox_gN(x->R(x) - vstar[i], dR, u[i], ϵ = ϵ, λ = α, l = -1e-14)
        end
        return x
end

# Takes a scalar i-v function and its derivative, and an offset vstar. Lower bound is currently
# hard coded for the Shockley equation.
# This version stores the result in the external variable ret.
@inline function res_diode_external!(u, R, dR, vstar, ret; α = 0.5, ϵ = 0.0001)
        Threads.@threads for i = 1:length(u)
                ret[i] = prox_gN(x->R(x) - vstar[i], dR, u[i], ϵ = ϵ, λ = α, l = -1e-14)
        end
        nothing
end

# α-resolvent of scalar function R, offset by ustar 
function res_scalar(u, ustar, R; α = 0.5, ϵ = 0.001)
        x = zeros(size(u))
        Threads.@threads for i = 1:length(u)
                x[i] = prox_l(x -> R(x) - ustar[i], u[i], ϵ = ϵ, λ = α)
        end
        return x

end

# discrete time implementation of the potassium current.  
function potassium(v, D)
        # old values, from one of Drion's papers
        # gK = 80e-3
        # vK = -80e-3
        # Hodgkin Huxley values:
        gK = 19e-3
        vK = 12e-3
        # backwards in time derivative

        # compute n
        n = (D + Diagonal(vec(α(v) .+ β(v))))\α(v)

        # compute i
        i = gK .* n.^4 .* (v .- vK)
end

# saturation
function sat(x::Float64; slope = 1, a = 1)
        if x > a
                return slope*a
        elseif x < 0
                #return -slope*a
                return 0
        else
                return slope*x
        end
end

function α(v)
        return 0.01 .*(10 .+ v)./(exp.((10 .+ v)./10) .- 1)
end

function β(v)
        return 0.125 .*exp.(v./80)
end
