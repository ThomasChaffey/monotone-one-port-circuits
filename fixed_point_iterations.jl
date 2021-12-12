## A collection of algorithm implementations for performing fixed point iterations
using Printf

# set up an exception for unstable schemes
struct UnstableException <: Exception
    var::String
end

Base.showerror(io::IO, e::UnstableException) = print("UnstableException ", io, e.var, " iterations" )

# conductance of scalar resistor
function G(v)
    return v^3
end
# compute the proximal operator of a scalar function using guarded Newton.
# λ is the resolvent parameter.  α is the guard parameter
function prox_gN(df, ddf, v::Float64; λ = 0.5, ϵ = 0.01, α = 0.5, l::Float64 = -Inf, u::Float64 = Inf)
    x::Float64 = v
    if v < l
        x = l + ϵ
    elseif v > u
        x = u - ϵ
    end

    # calculate initial u, l, x
    g::Float64 = df(x) + (1/λ)*(x - v)
    a::Float64 = x - λ*g
    b::Float64 = x
    if g < 0
        a = x
        b = x - λ*g
    end
    l = l > a ? l : a #maximum([l, a])
    u = u > b ? b : u #minimum([u, b])
    x = (l + u)/2
    while u - l > ϵ
        αl = (u + l)/2 - α*(u - l)/2
        αu = (u + l)/2 + α*(u - l)/2
        # if x < 0
        #     print(v)
        # end
        dϕ = df(x) + (1/λ)*(x - v)
        ddϕ = ddf(x) + 1/λ
        update = x - dϕ/ddϕ
        # project onto interval
        if update < αl
            x = αl
        else x > αu
            x = αu
        end
        # the rest is the same as the localisation algorithm
        g = df(x) + (1/λ)*(x - v)
        a = x - λ*g
        b = x
        if g < 0
            a = x
            b = x - λ*g
        end
        l = l > a ? l : a #l = maximum([l, a])
        u = u > b ? b : u #u = minimum([u, b])
        x = (l + u)/2
    end
    return x
end


function forward_backward(A, y0, RB; debug = false, ϵ = 0.001, M = 1e4, α = 0.05, verbose=true)
    # ϵ is the relative convergence tolerance. α is the resolvent parameter.  M is the definition of instability. A is evaluated forward. RB is the α resolvent of B.
    count = 0
    while true
        count += 1
        y1 = y0
        ytemp = y0 - α*A(y0);
        y0 = RB(ytemp, α)
        if maximum(abs.(y0 - y1)) < ϵ #/maximum(abs.(y0)) < ϵ
            break
        elseif maximum(abs.(y0)) > M
            throw(UnstableException(string(count)))
        end
        if count%100 == 0 && verbose
            @printf("Count: %d Absolute tolerance: %0.6f\n", count,  maximum(abs.(y0 - y1)))
        end
        if count%100 == 0 && debug
                p = plot(LinRange(0, 1, length(y0)), y0)
                display(p)
        end
    end
    return y0
end

# Douglas-Rachford splitting.  First two arguments are resolvents. i is the initial guess.
function Douglas_Rachford(R1, R2, i; ϵ = 1e-3, M = 1e4, name = "", iters = 0, debug=false, verbose = true)
    count = 0
    i0 = copy(i)

    while true
        count += 1
        i1 = copy(i0) # this might be quite a slow operation

        # first resolvent
        x_half = R1(i0)

        z_half = 2*x_half - i0
        
        # second resolvent
        x = R2(z_half)

        i0 = i0 + x - x_half
        
        abstol = maximum(abs.(i0 - i1)) 
        reltol = maximum(abs.(i0)) == 0 ? abstol : maximum(abs.(i0 - i1))/maximum(abs.(i0)) 
        if abstol < ϵ
            break
        elseif abstol > M
            throw(UnstableException(string(count," ", name)))
        end

        if count%100 == 0 && verbose
            @printf("DR Iterations: %d\nAbsolute tolerance: %.15f\nRelative tolerance: %.5f\n", count, abstol, reltol)
        end

        if iters > 0 && count > iters
                break
        end

        if count%100 == 0 && debug
                p = plot(LinRange(0, 1, length(i0)), i0, label = "DR output")#, ylims=(-1, 1) )
                display(p)
        end
    end
    return i0
end

# differentiation and integration
function diff(v; periods = 1)
    N = length(v)
    Tint = trunc(Int, N/periods) # number of entries per period
    # differentiate q to get i
    D = Array(Bidiagonal(vec(ones(1, N)), -1*vec(ones(1, N - 1)), :L))
    D[1, end] = -1
    D = Tint*D # divide by dt
    return D*v
end

function integrate(v; periods = 1)
    N = length(v)
    Tint = trunc(Int, N/periods) # number of entries per period
    J = (1/Tint)*LowerTriangular(ones(N, N))
    return J*v
end
