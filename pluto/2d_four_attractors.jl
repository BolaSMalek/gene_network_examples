### A Pluto.jl notebook ###
# v0.14.5

using Markdown
using InteractiveUtils

# ╔═╡ d4260fcc-784e-4341-ae84-c7e8ab288c4b
using LaTeXStrings, DifferentialEquations, LinearAlgebra, SparseArrays

# ╔═╡ 82272c06-3861-4aed-a08d-bed2e84570fc
begin
	using Plots
	gr()
end

# ╔═╡ b91fd324-adf3-11eb-36e5-59bf4de5e0f9
md"# 2D Multiple Attractor System"

# ╔═╡ 41f97746-5818-410f-b3cb-e59aa9a5bc99
md"_Setup vector field equations. Can't split into jump process._"

# ╔═╡ 5a0a71b5-0b18-42be-b567-c01feb0d9af6
begin
	α = 0.1
	β = 3
	n = 4
	dfx(x, y) = 9x + 9y - (1 + 2x^3 + 2y^3)
	dfy(x, y) = 1 + 2x^3 + 11y - (11x + 2y^3)
end

# ╔═╡ f56451d1-6f67-494b-82b0-2369558d6819
begin
	n_min = -3
	n_max = 3
end

# ╔═╡ a342ae72-b363-4142-929c-78e682871d81
md"_Compute the vector field_"

# ╔═╡ 7cb95f1f-8a19-484c-a33f-a13348dce332
begin
	dq = (n_max - n_min) / 10.
	xs = n_min-1:dq:n_max+1
	ys = n_min-1:dq:n_max+1
	
	dxs = n_min-1:.05:n_max+1
	dys = n_min-1:.05:n_max+1

	df(x, y) = normalize([dfx(x, y), dfy(x,y)]) ./ 1.75

	xxs = [x for x in xs for y in ys]
	yys = [y for x in xs for y in ys]

	quiver(xxs, yys, quiver=df)
	contour!(dxs, dys, dfx, levels=[0], color=:orange, label="X Nullcline", colorbar = false)
	p1 = contour!(dxs, dys, dfy, levels=[0], color=:green, label = "Y Nullcline", colorbar = false)
	p1
end

# ╔═╡ 7d1e052e-c9b6-4fa3-b4aa-980e60e53961
md"_Setup the Stochastic Differential Equation model with diagonal noise_"

# ╔═╡ 601e7f8b-efa8-4710-b088-78b825b0454a
function f(du, u, p, t)
	du[1] = dfx(u...)
	du[2] = dfy(u...)
	du[1], du[2]
end

# ╔═╡ e3b32bba-b8c8-48ba-8918-c73f1f1e9db9
begin
	g(du, u, p, t) = begin
	  du[1] = 1.25
	  du[2] = 1.25
	end
	sde_prob= SDEProblem(f, g, [0., 0.], (0.0, 1000.0))
	sde_sol = solve(sde_prob, SRIW1())
end

# ╔═╡ 67d5512b-fe45-47d7-9b18-c847a8010577
pstoch_1 = plot(sde_sol,title="Stochastic Simulation", labels=[L"x" L"y"])

# ╔═╡ a1d79d06-afdd-4097-94ac-716779173e6c
pstoch_2 = plot(sde_sol[1000:end], vars = [(1,2)], label="", xlabel=L"x", ylabel = L"y", title="Time Evolution State Plot")

# ╔═╡ eef51429-8458-421a-8366-35577019e204
# pstoch = plot(pstoch_1, pstoch_2, layout=(1, 2), size=(900, 350))

# ╔═╡ 10c9d45f-d8ab-4acd-88ba-8a80a747e30f
# savefig(pstoch, "ch3_four_attractor_stoch.pdf")

# ╔═╡ 9e084100-e5d1-488c-8eaf-769711af1d95
md"_Helper function to tip towards different quadrants for each run of the simulation_"

# ╔═╡ dc5e6e4a-93d6-47c3-80ef-8d22d3483e99
function get_u0(i)
	offset = 1.0
	new_u0 = [0.0, 0.0]
	if i % 4 == 0
		new_u0 .+= [offset, offset]
	elseif i % 4 == 1
		new_u0 .+= [-1* offset, offset]
	elseif i % 4 == 2
		new_u0 .+= [-1* offset, -1* offset]
	else
		new_u0 .+= [offset,-1* offset]
	end
	new_u0
end

# ╔═╡ 3481f158-573b-4ca4-9f2f-fa0e4b68edb4
md"_Add smaller noise term for longer time simulation._

Note that this is equivalent to modifying `D` in the Fokker-Plank Equations and yeilds different probability distributions."

# ╔═╡ 25b8d3e0-4402-4e84-91a0-c5d1c4684ac4
begin 
	g2(du, u, p, t) = begin
	  du[1] = 0.5
	  du[2] = 0.5
	end
end

# ╔═╡ 1870a27b-683b-463a-b5cc-9ac073110091
md"_Helper to compute the 2D histograms for unnomalized probability_"

# ╔═╡ c1b075dc-c4dc-4bd9-bbd9-bbb0119fc54a
begin
	nbins = 257
	bin_min = -10.
	bin_max = 10.
	xbins = range(bin_min, bin_max, length=nbins)
	ybins = range(bin_min, bin_max, length=nbins)
	xbins
	
	function bucket_idx(val, vec)
		idx = 1
		while val > vec[idx]
			idx += 1
		end
		idx - 1 
	end
	
	function althist(x, y, dt, xedges, yedges) 
	   counts = spzeros(length(xedges)-1, length(yedges)-1)

		for i=1:length(x) - 1
			r = bucket_idx(x[i], yedges)
			c = bucket_idx(y[i], xedges)
			counts[r,c] += dt[i]
		end

 		counts
	end

	drop_num = 2000
end

# ╔═╡ 351e030b-0d6c-42ea-8c5b-9cad2af5c12c
unnorm = spzeros(length(xbins)-1, length(ybins)-1)

# ╔═╡ 8b509897-8b02-4af7-a0bd-990f2cb3df87
md"_Rerun the following cell to add more data points to the simulation._"

# ╔═╡ 75bcc971-1a89-4dd4-a5ba-a1397f5db274
begin
	for i in 1:4*100
		u00 = get_u0(i)
		sde_prob_long = SDEProblem(f, g2, u00, (0.0, 1e3))
		sde_sol_long= solve(sde_prob_long, SRIW1())
		unnorm .+= althist(sde_sol_long[1, drop_num:end], sde_sol_long[2, drop_num:end], diff(sde_sol_long.t[drop_num:end]), xbins, ybins)
	end
	unnorm
end

# ╔═╡ 86019cdf-c7de-4495-ba61-de0852f2b496
md"_Normalized and visualize the probability and resulting Quasi-potential._"

# ╔═╡ 13347146-e598-4d25-9d92-4b047a9de492
prob = unnorm ./ sum(unnorm)

# ╔═╡ 6c523287-b8db-4ef7-a08a-f61bf2639278
begin
	plotly()
	p = surface(- log.(prob))
	gr()
	p
end

# ╔═╡ bc84c949-6b26-4e02-84bf-a2093508a0dc
begin
	plotly()
	pp = surface(xbins, ybins,prob,c = :grayC)
	gr()
	pp
end

# ╔═╡ 6af8e6ef-2444-4c88-ad81-a645b8971527
begin
	xplot = range(bin_min, bin_max, length=nbins-1)
	yplot = range(bin_min, bin_max, length=nbins-1)
end

# ╔═╡ b3a3b34b-ffff-4b35-952e-d844b5830ce7
 begin
	pp1 = surface(xplot, yplot,-log.(prob), c = :thermal)
	pp2 =  surface(xplot, yplot, prob, c = :grayC)
	ps = plot(pp1, pp2, layout =(2, 1), size=(500, 800))
end

# ╔═╡ 4e4f16b1-7c8a-4435-91f9-6a0d4ff54b22
# savefig(ps, "ch3_4a_ad_hoc.pdf")

# ╔═╡ c37e269f-ac80-4bd1-81d0-e5a476e85272
md"_Helper function to generate -∇ ² to solve the Helmhotz equation_"

# ╔═╡ 164f3c77-5e95-4feb-8783-54c381f4b184
begin
	import DiffEqOperators
	const DiffEqOp = DiffEqOperators

	eye(N, M) = sparse(I, N, M)
	eye(M) = eye(M, M)
	diff1(M) = [ [1.0 zeros(1, M - 1)]; diagm(1 => ones(M - 1)) - eye(M) ]
	sdiff1(M) = sparse(diff1(M))
	flatshape(X) = reshape(X, length(X))


	# make the discrete -Laplacian in 2d, with Dirichlet boundaries
	function laplacian2d(Nx, Ny, Lx, Ly)
		dx = Lx / (Nx + 1)
		dy = Ly / (Ny + 1)
		Dx = sdiff1(Nx) / dx
		Dy = sdiff1(Ny) / dy
		Ax = Dx' * Dx
		Ay = Dy' * Dy
		return Dx, Dy, kron(eye(Ny), Ax) + kron(Ay, eye(Nx))
	end
end

# ╔═╡ c0960c8a-76d0-466a-889b-2179d02518a7
# fg =  plot(p1, p2, p3, layout=(3,1), size=(750, 2250))

# ╔═╡ eacf4f60-956d-4408-b67d-58465a0411dc
# savefig(fg, "ch3_four_attractor.pdf")

# ╔═╡ 23bd055e-599b-407a-9918-d9880d257db9
md"### Appendix: Helper decomposition functions"

# ╔═╡ 3fd9492a-d75e-4603-bafc-4e48fd422385
## We are trying to solve the following equation 
## ∇²Uᴴ (x⃗) = -∇ ⋅ F(x⃗)
## We are starting in 2 dimensions
function helmholtz(prob::DiscreteProblem, minval::Int64, maxval::Int64, nsample::Int64) 
    dx = (maxval - minval) / (nsample + 1)
    dy = (maxval - minval) / (nsample + 1)
    xs = range(minval, maxval, length=nsample)
    ys = range(minval, maxval, length=nsample)
    ∂x(x, y) = prob.f.f([0.0, 0.0], [x, y], prob.p, nothing)[1]
    ∂y(x, y) = prob.f.f([0.0, 0.0], [x, y], prob.p, nothing)[2]
    # Let's first construct F = [f1, f2]
    f1 = [∂x(x, y) for x in xs, y in ys]
    f2 = [∂y(x, y) for x in xs, y in ys]
    _, _, Δ = laplacian2d(nsample, nsample, maxval - minval, maxval - minval)

    Dx = DiffEqOp.CenteredDifference{1}(1, 2, dx, nsample)
    Dy = DiffEqOp.CenteredDifference{2}(1, 2, dy, nsample)

    bcx = DiffEqOp.Dirichlet0BC{1}(eltype(f1), size(f1))
    bcy = DiffEqOp.Dirichlet0BC{2}(eltype(f2), size(f2))

    rhs =  Dx * bcx * f1 + Dy * bcy *  f2
    
    u = Δ \ flatshape(rhs)
    xs, ys, reshape(u, size(rhs)...)
end

# ╔═╡ 751ac213-1312-4f0e-bdd8-fa2a10c4554f
begin
	nsample = 512
	dprob = DiscreteProblem(f, [0.0,0.0], (0, 100.), nothing)
	x, y, u_h = helmholtz(dprob, -3, 3, nsample)
end

# ╔═╡ 8101d876-4c69-43e2-8492-59a799707151
begin
	gr()
	fg = surface(x, y, u_h, colorbar=:none,  c=:viridis, aspect_ratio=:equal, xlabel=L"x", ylabel=L"y", zlabel=L"U^H")
end

# ╔═╡ b6ce865e-e1d0-47cd-bbfc-e144459d92a2
p3 = heatmap(x, y, u_h, c=:viridis)

# ╔═╡ 75902190-09c8-41aa-b3ab-ceeaee4ac821

function make_inplace(f::Function)
    function inplace_f(out, x)
        out .= f(x)
    end
    return inplace_f
end


# ╔═╡ b57411c3-a92d-4ce6-82e0-ad0cc4b56f3f
function normal(prob::DiscreteProblem, minval::Int64, maxval::Int64, nsample::Int64, u0::Matrix{Float64})
    dx = (maxval - minval) / (nsample + 1)
    dy = (maxval - minval) / (nsample + 1)

    xs = range(minval, maxval, length=nsample)
    ys = range(minval, maxval, length=nsample)
    ∂x(x, y) = prob.f.f([0.0, 0.0], [x, y], prob.p, nothing)[1]
    ∂y(x, y) = prob.f.f([0.0, 0.0], [x, y], prob.p, nothing)[2]
    
    f1 = [∂x(x, y) for x in xs, y in ys]
    f2 = [∂y(x, y) for x in xs, y in ys]
    
    Dx = DiffEqOp.UpwindDifference{1}(1, 2, dx, nsample, 1)
    Dy = DiffEqOp.UpwindDifference{2}(1, 2, dy, nsample, 1)

    bcx = DiffEqOp.Dirichlet0BC{1}(eltype(u0), size(u0))
    bcy = DiffEqOp.Dirichlet0BC{2}(eltype(u0), size(u0))

    function find_my_zero(u)
        dudx = Dx * bcx * u
        dudy = Dy * bcy * u
        return dot(dudx, f1 .+ dudx) + dot(dudy, f2 .+ dudy)
    end

    fmz! = make_inplace(find_my_zero)

    sol = nlsolve(fmz!, u0, show_trace=true, iterations=3000)
    sol.zero
end


# ╔═╡ Cell order:
# ╟─b91fd324-adf3-11eb-36e5-59bf4de5e0f9
# ╠═d4260fcc-784e-4341-ae84-c7e8ab288c4b
# ╠═82272c06-3861-4aed-a08d-bed2e84570fc
# ╟─41f97746-5818-410f-b3cb-e59aa9a5bc99
# ╠═5a0a71b5-0b18-42be-b567-c01feb0d9af6
# ╠═f56451d1-6f67-494b-82b0-2369558d6819
# ╟─a342ae72-b363-4142-929c-78e682871d81
# ╠═7cb95f1f-8a19-484c-a33f-a13348dce332
# ╟─7d1e052e-c9b6-4fa3-b4aa-980e60e53961
# ╠═601e7f8b-efa8-4710-b088-78b825b0454a
# ╠═e3b32bba-b8c8-48ba-8918-c73f1f1e9db9
# ╠═67d5512b-fe45-47d7-9b18-c847a8010577
# ╠═a1d79d06-afdd-4097-94ac-716779173e6c
# ╠═eef51429-8458-421a-8366-35577019e204
# ╠═10c9d45f-d8ab-4acd-88ba-8a80a747e30f
# ╟─9e084100-e5d1-488c-8eaf-769711af1d95
# ╠═dc5e6e4a-93d6-47c3-80ef-8d22d3483e99
# ╟─3481f158-573b-4ca4-9f2f-fa0e4b68edb4
# ╠═25b8d3e0-4402-4e84-91a0-c5d1c4684ac4
# ╟─1870a27b-683b-463a-b5cc-9ac073110091
# ╠═c1b075dc-c4dc-4bd9-bbd9-bbb0119fc54a
# ╠═351e030b-0d6c-42ea-8c5b-9cad2af5c12c
# ╟─8b509897-8b02-4af7-a0bd-990f2cb3df87
# ╠═75bcc971-1a89-4dd4-a5ba-a1397f5db274
# ╟─86019cdf-c7de-4495-ba61-de0852f2b496
# ╠═13347146-e598-4d25-9d92-4b047a9de492
# ╠═6c523287-b8db-4ef7-a08a-f61bf2639278
# ╠═bc84c949-6b26-4e02-84bf-a2093508a0dc
# ╠═6af8e6ef-2444-4c88-ad81-a645b8971527
# ╠═b3a3b34b-ffff-4b35-952e-d844b5830ce7
# ╠═4e4f16b1-7c8a-4435-91f9-6a0d4ff54b22
# ╟─c37e269f-ac80-4bd1-81d0-e5a476e85272
# ╠═164f3c77-5e95-4feb-8783-54c381f4b184
# ╠═751ac213-1312-4f0e-bdd8-fa2a10c4554f
# ╠═8101d876-4c69-43e2-8492-59a799707151
# ╠═b6ce865e-e1d0-47cd-bbfc-e144459d92a2
# ╠═c0960c8a-76d0-466a-889b-2179d02518a7
# ╠═eacf4f60-956d-4408-b67d-58465a0411dc
# ╟─23bd055e-599b-407a-9918-d9880d257db9
# ╟─3fd9492a-d75e-4603-bafc-4e48fd422385
# ╟─75902190-09c8-41aa-b3ab-ceeaee4ac821
# ╟─b57411c3-a92d-4ce6-82e0-ad0cc4b56f3f
