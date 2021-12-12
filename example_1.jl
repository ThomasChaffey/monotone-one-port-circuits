## envelope detector inverse
using Printf
using Plots
using LinearAlgebra
using DataFrames
using CSV

include("fixed_point_iterations.jl")
include("components.jl")

struct UnstableException <: Exception
    var::String
end

# exception for unstable iteration
Base.showerror(io::IO, e::UnstableException) = print(io, e.var, " iterations" )

# Douglas-Rachford splitting
function envelope_detector_inverse_dr(i, vstar, R_RC; ϵ = 0.01, M = 1e4, α = 0.5)
    count = 0
    i0 = copy(i)
    x_half = zeros(size(i0))

    while true
        count += 1
        i1 = copy(i0)
        # resolvent of RC circuit - in-place code must faster
        # x_half = R_RC*i0
        mul!(x_half, R_RC, i0)

        z_half = 2*x_half - i0
        x = zeros(size(z_half))

        # resolvent of diode (elements defined in "components.jl")
        for i = 1:length(z_half)
            x[i] = prox_gN(x -> R_diode(x) - vstar[i], dR_diode, z_half[i], ϵ = ϵ, λ = α, l = -1e-14)
        end

        i0 = i0 .+ x .- x_half
        
        if maximum(abs.(i0 - i1))/maximum(abs.(i0)) < ϵ
            break
        elseif maximum(abs.(i0 - i1)) > M
            throw(UnstableException(string(count)))
        end

        if count%100 == 0
            @printf("Iterations: %d\n", count)
        end

    end
    return i0
end

## run the example
function example_1()
    N = 500
    T = LinRange(0, 1, N)
    vstar = sin.(T*2*pi)
    i0 = T
    # forward linear operator for a parallel RC circuit
    α = 0.01
    R = R_RC(N, α)
    istar = envelope_detector_inverse_dr(i0, vstar, R, ϵ=0.00001, α=α)
     
    # plot
    p1 = plot(T, vstar, label = "v - input", ylabel = "V")
    p2 = plot(T, istar, label = "i - output", ylabel = "A", xlabel = "Time (s)")
    plot(p1, p2, layout = (2, 1))
    #savefig("envelope_detector_inverse.pdf")

    # save the data
    #d = DataFrame(t = T, v = vstar, i = istar)
    #CSV.write("envelope_detector_inverse.csv", d)
end

## run the check 
example_1()
