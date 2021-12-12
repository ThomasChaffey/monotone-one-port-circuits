using Printf
using Plots
using LinearAlgebra
using DataFrames
using CSV
using Random
using Distributions

include("fixed_point_iterations.jl")
include("components.jl")


# resolvent of a conductor
function RG(i, α, istar)
        # conductance, mohs
        G = 0.002
        return (i + α*istar)/(α*G + 1) 
end

# golbal stores old value of inverse, for use in nested iterations
global old_G1_inverse
function G1_inverse(i, y0, α, ϵ, D)
        y_out = forward_backward(u -> potassium(u, D), y0, (y, α) -> RG(y, α, i), α = α, ϵ = ϵ, verbose=false)
        old_G1_inverse = copy(y_out)
end

# potassium current in series with resistance
function example_3()
        # signal parameters
        N = 256 # must be divisible by periods
        periods = 1
        T = LinRange(0, 1, N)
        istar = sin.(periods*T*2*pi)

        # pre-compute derivative for speed
        D = Array(Bidiagonal(vec(ones(1, N)), -1*vec(ones(1, N - 1)), :L))
        D[1, end] = -1
        D = D*N # divide by dt
        # RC resolvents, standard values
        α = 0.5
        ϵ = 0.001
        i0 = T
        vstar = G1_inverse(istar, i0, α, ϵ, D) 
        p = plot(vstar, istar, ylabel = "Lissajous figure")
        #d = DataFrame(t = T, i = istar, v = vstar)
        #CSV.write("K_current.csv", d)
end

example_3()
