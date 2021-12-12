using Printf
using Plots
using DataFrames
using CSV
include("components.jl")

# length n circuit
function n_envdet(n)
        # number of time steps
        N = 256
        # make all the α equal.  
        α = 1.5

        ϵ = 0.0001
        M = 1e3
        count = 0

        T = LinRange(0, 1, N)
        vstar::Array{Float64, 1} = 1 .+ sin.(T*2*pi)

        # Signal vectors.  Index as v_new[:, k].
        v_new::Array{Float64, 2} = repeat(vstar, 1, n)
        i_new::Array{Float64, 2} = repeat(vstar, 1, n)
        v_old::Array{Float64, 2} = repeat(vstar, 1, n)
        i_old::Array{Float64, 2} = repeat(vstar, 1, n)

        # array for storing matrix multiplications
        temp::Array{Float64, 1} = zeros(size(vstar))
        temp2::Array{Float64, 1} = zeros(size(vstar))
        temp3::Array{Float64, 1} = zeros(size(vstar))

        # RC circuit (resistance form)
        G1inv = RC(N)
        # α-resolvent of RC (conductance form)
        # have to offset by using JRC*x + JRC*α*in
        JRC = S_GC(N, α)

        while true
                count+=1

                # All matrix operations here are done using BLAS routines in-place, much faster!
                # Hard to read though...  see the function double_envdet() for a more readable function.
                mul!(temp, G1inv, @view i_old[:, 1])
                BLAS.scal!(length(temp), α, temp, 1)
                @views axpby!(1.0, i_old[:, 1], -1.0, temp)
                @views res_diode_external!(temp, R_diode, dR_diode, v_old[:, 1], i_new[:, 1], α = α, ϵ = ϵ)
                for k = 2:n
                        mul!(temp, JRC, i_old[:, k])#JRC*i_old[:, k]
                        BLAS.scal!(length(temp), α, temp, 1)
                        @views BLAS.blascopy!(length(i_new[:, k-1]), i_new[:, k-1], 1, temp2, 1)
                        @views axpby!(1.0, v_old[:, k-1], -α, temp2)
                        mul!(temp3, JRC, temp2)
                        axpby!(1.0, temp3, 1.0, temp)
                        @views BLAS.blascopy!(length(temp), temp, 1, v_new[:, k-1], 1)

                        @views BLAS.blascopy!(length(v_new[:, k-1]), v_new[:, k-1], 1, temp, 1)
                        @views axpby!(1.0, i_old[:, k], -α, temp)
                        @views res_diode_external!(temp, R_diode, dR_diode, v_new[:, k], i_new[:, k],  α = α, ϵ = ϵ)
                end

                maxnorm = maximum([norm(i_old[:, k] - i_new[:, k]) for k = 1:n])
                if maxnorm < ϵ 
                        break
                elseif maxnorm > M
                        print("unstable")
                        break
                end


                if count%1000 == 0
                        @printf("%d iterations, Δi2: %.5f\n", count, maxnorm)
                        p = plot(T, i_new[:, n])
                        display(p)
                end

                copy!(i_old, i_new)
                copy!(v_old, v_new)

        end
        #d = DataFrame(t = T, v = vstar, i = i_new[:, n])
        #CSV.write("large_scale_100k.csv", d)
end

# example with only two repeated units.  Not optimized for speed, but more readable.
function double_envdet()
        N = 512
        α0 = 1.5
        α1 = 1.5
        α2 = 1.5

        ϵ = 0.0001
        M = 1e3
        count = 0

        T = LinRange(0, 1, N)
        vstar = 1 .+ sin.(T*2*pi)
        i2old = zeros(size(vstar))
        i1old = zeros(size(vstar))
        v1old = zeros(size(vstar))

        # RC circuit (resistance form)
        G1inv = RC(N)
        # α0-resolvent of RC (conductance form)
        # have to offset by i2 using JRC*x + JRC*α1*i2
        JRC = S_GC(N, α1)

        while true
                count+=1

                i1new .= res_diode(i1old .- α0*G1inv*i1old, R_diode, dR_diode, v1old, α = α0, ϵ = ϵ)
                v1new = JRC*(v1old - α1*i1new) + JRC*α1*i2old
                i2new = res_diode(i2old .- α2*v1new, R_diode, dR_diode, vstar, α = α2, ϵ = ϵ)

                maxnorm = maximum([norm(v1old - v1new), norm(i1old - i1new), norm(i2old - i2new)])
                if maxnorm < ϵ 
                        break
                elseif maxnorm > M
                        print("unstable")
                        break
                end


                if count%1000 == 0
                        @printf("%d iterations, Δi2: %.5f, Δi1: %.5f, Δv1: %.5f\n", count, 
                                norm(i2new - i2old), norm(i1new - i1old), norm(v1new - v1old))
                        p = plot(T, i2new)
                        display(p)
                end

                i2old = i2new
                i1old = i1new
                v1old = v1new

        end
end

n_envdet(2)
