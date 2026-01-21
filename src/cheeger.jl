# ============================================================
# Solve the λ_μ program on a graph given by adjacency matrix A
# Requires: JuMP, Ipopt
# ============================================================

using LinearAlgebra
using SparseArrays
using Random
using JuMP
using Ipopt
using DelimitedFiles
using MatrixNetworks
using Gurobi

function load_graph_txt(path2file::String)
    E = readdlm(path2file, Int64)
    I = E[:, 1]
    J = E[:, 2]
    A = sparse(I, J, ones(Int64, length(I)))
    A, _ = largest_component(A)
    return A
end

"""
    solve_lambda_mu(A, μ; starts=20, seed=1, ipopt_print_level=0)

Solve:
    minimize x' L x
    s.t.    x' d = 0
            x' D x = 1
            |x_i| <= a  for all i
            |x_i| >= b  for all i

where:
    D = diag(degrees), L = D - A, d = degrees vector
    Vol(G) = sum(d)
    a = sqrt((1-μ)/(μ*Vol(G)))
    b = sqrt(μ/((1-μ)*Vol(G)))

Returns: (best_obj, best_x, info_dict)

Notes:
- Nonconvex due to |x_i| >= b, so solver is local.
- We do multi-start and keep the best feasible solution.
- Feasibility requires b <= a, which implies μ <= 0.5 (given these formulas).
"""
function solve_lambda_mu(A::AbstractMatrix{T}, μ::Real;
                         starts::Int=20, seed::Int=1, ipopt_print_level::Int=0) where {T<:Real}

    n = size(A, 1)
    @assert size(A, 2) == n "A must be square"
    @assert 0 < μ < 1 "μ must be in (0,1)"

    # Build degree vector, D, Laplacian L
    d = vec(sum(A, dims=2))     
    VolG = sum(d)

    # Constants from the screenshot
    a = sqrt((1 - μ) / (μ * VolG))
    b = sqrt(μ / ((1 - μ) * VolG))

    @show a, b
    if b > a + 1e-12
        error("Infeasible bounds: lower |x_i| >= b is larger than upper |x_i| <= a. " *
              "This happens when μ > 0.5 with these formulas. Got a=$a, b=$b.")
    end

    # Use sparse L if possible
    Dmat = spdiagm(0 => d)
    L = Dmat - sparse(A)

    Random.seed!(seed)

    best_obj = Inf
    best_x = nothing
    best_status = nothing

    # Helper: create a good-ish starting point satisfying x'd≈0 and x'Dx≈1
    function make_start()
        x0 = zeros(Float64, n)
        for i in 1:n
            s = rand(Bool) ? 1.0 : -1.0
            # magnitude uniform in [b, a]
            x0[i] = s * (b + (a - b) * rand())
        end
        # enforce x' d = 0 approximately by subtracting projection onto d
        denom = dot(d, d)
        if denom > 0
            x0 .-= (dot(d, x0) / denom) .* d
        end
        # renormalize to satisfy x' D x = 1 approximately
        normD = sqrt(sum(d .* (x0 .^ 2)))
        if normD > 0
            x0 ./= normD
        end
        # clip into [-a, -b] ∪ [b, a] to respect bounds for a start
        for i in 1:n
            xi = x0[i]
            if abs(xi) < b
                x0[i] = sign(xi == 0 ? (rand(Bool) ? 1.0 : -1.0) : xi) * b
            elseif abs(xi) > a
                x0[i] = sign(xi) * a
            end
        end
        return x0
    end

    for s in 1:starts
        model = Model(Ipopt.Optimizer)
        set_silent(model)
        set_optimizer_attribute(model, "print_level", ipopt_print_level)
        set_optimizer_attribute(model, "max_iter", 5000)
        set_optimizer_attribute(model, "tol", 1e-8)

        @variable(model, x[1:n])

        # Bounds implementing ||x||_∞ ≤ a  and  |x_i| ≥ b  (as quadratic inequalities)
        # -a ≤ x_i ≤ a + also enforce x_i^2 ≥ b^2 (nonconvex)
        for i in 1:n
            @constraint(model, -a <= x[i] <= a)
            @constraint(model, x[i]^2 >= b^2)
        end

        # Orthogonality: x' d = 0
        @constraint(model, d' * x == 0)

        # Normalization: x' D x = 1  i.e., sum(d_i x_i^2) = 1
        @constraint(model, x' * Diagonal(d) * x == 1)

        # Objective: x' L x
        @objective(model, Min, x' * L * x)

        x0 = make_start()
        for i in 1:n
            set_start_value(x[i], x0[i])
        end

        optimize!(model)

        term = termination_status(model)
        primal = primal_status(model)

        @show term, primal

        if (term == MOI.LOCALLY_SOLVED || term == MOI.OPTIMAL) && primal == MOI.FEASIBLE_POINT
            xsol = value.(x)
            objval = objective_value(model)
            if objval < best_obj
                best_obj = objval
                best_x = xsol
                best_status = (term, primal)
            end
        end
    end

    info = Dict(
        "μ" => μ,
        "a" => a,
        "b" => b,
        "VolG" => VolG,
        "starts" => starts,
        "best_status" => best_status,
    )

    if best_x === nothing
        error("No feasible solution found in $starts starts. Try increasing `starts`, " *
              "or loosening tolerances, or check feasibility of your instance.")
    end

    return best_obj, best_x, info
end


"""
    solve_phi_mu(A, μ; time_limit=300.0)

Solve:
  ϕ_μ(G) = min_S ϕ(S)
  s.t. μVol(G) ≤ Vol(S) ≤ (1-μ)Vol(G)

where ϕ(S) = cut(S, V\\S) / min(Vol(S), Vol(V\\S)),
Vol(S) = sum_i d_i * y_i, d_i = sum_j A[i,j], and y_i ∈ {0,1}.

Returns: (best_phi, y_solution, info::Dict)
"""
function solve_phi_mu(A::AbstractMatrix{<:Real}, μ::Real; time_limit::Real=300.0)
    n = size(A, 1)
    @assert size(A, 2) == n "A must be square"
    @assert 0 <= μ <= 1/2 "μ must be in [0,1/2]"
    @assert issymmetric(A) "This formulation assumes an undirected graph (A symmetric)"
    @assert all(A .>= 0) "Adjacency weights must be nonnegative"

    d = vec(sum(A, dims=2))
    VolG = sum(d)
    @assert VolG > 0 "Vol(G) must be positive"

    # Build edge list from upper triangle nonzeros (i < j)
    edges_i = Int[]
    edges_j = Int[]
    weights = Float64[]
    for i in 1:n
        for j in (i+1):n
            w = float(A[i, j])
            if w != 0.0
                push!(edges_i, i)
                push!(edges_j, j)
                push!(weights, w)
            end
        end
    end
    m = length(weights)
    @assert m > 0 "Graph has no edges (cut is always 0; objective is degenerate)."

    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "TimeLimit", time_limit)

    # Required for bilinear (nonconvex) quadratic constraints in Gurobi:
    set_optimizer_attribute(model, "NonConvex", 2)

    @variable(model, y[1:n], Bin)        # membership of S
    @variable(model, z[1:m] >= 0)        # z_e >= |y_i - y_j|, (continuous is fine)
    @variable(model, v >= 0)             # Vol(S)
    @variable(model, c >= 0)             # cut(S)
    @variable(model, t >= 0)             # objective value (conductance)

    # Volume definition and constraints
    @constraint(model, v == sum(d[i] * y[i] for i in 1:n))
    @constraint(model, μ * VolG <= v)
    @constraint(model, v <= VolG / 2)

    # Linearize |y_i - y_j| with z_e >= y_i - y_j and z_e >= y_j - y_i
    # Also z_e <= 1 is enough since y are binary
    for e in 1:m
        i = edges_i[e]
        j = edges_j[e]
        @constraint(model, z[e] >= y[i] - y[j])
        @constraint(model, z[e] >= y[j] - y[i])
        @constraint(model, z[e] <= 1)
    end

    # cut(S) = Σ w_e z_e
    @constraint(model, c == sum(weights[e] * z[e] for e in 1:m))

    # Enforce t >= c / min(v, VolG - v) via bilinear inequalities:
    @constraint(model, c <= t * v)

    @objective(model, Min, t)

    optimize!(model)

    term = termination_status(model)
    if term != MOI.OPTIMAL && term != MOI.TIME_LIMIT && term != MOI.SUBOPTIMAL
        error("Solver did not return a usable solution. termination_status = $term")
    end

    ysol = value.(y)
    best_phi = value(t)

    info = Dict(
        "termination_status" => term,
        "VolG" => VolG,
        "VolS" => value(v),
        "cut"  => value(c),
        "μ"    => μ,
    )
    return best_phi, ysol, info
end

# -----------------------
# Example usage:
# -----------------------
# Toy example: cycle graph on 6 nodes
n = 6
A = zeros(Float64, n, n)
for i in 1:n
    A[i, mod1(i+1, n)] = 1
    A[i, mod1(i-1, n)] = 1
end

μ = 0.1
best_obj, best_x, info = solve_lambda_mu(A, μ; starts=30, seed=42)

println("Best objective: ", best_obj)
println("Best x: ", best_x)
println("Info: ", info)

A = load_graph_txt("data/graph-2018-11-09.edges")
A == A'

μ = 0.45
best_obj, best_x, info = solve_lambda_mu(A, μ; starts=30, seed=42)
println("Best objective: ", best_obj)
println("Best x: ", best_x)
println("Info: ", info)

best_phi, ysol, info = solve_phi_mu(A, μ; time_limit=60.0)
println("Best phi: ", best_phi)
println("Best y: ", ysol)
println("Info: ", info)