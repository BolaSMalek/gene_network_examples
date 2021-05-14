### A Pluto.jl notebook ###
# v0.14.5

using Markdown
using InteractiveUtils

# ╔═╡ 193ee450-7b32-4394-9401-7b91bce21f31
using LaTeXStrings, DifferentialEquations, LinearAlgebra, Plots, SparseArrays

# ╔═╡ 385b2bda-24d8-464c-9257-c4393a0b1c29
using ForwardDiff

# ╔═╡ 13137d52-2ede-4a63-8eac-43b13def7466
md"# Self Activating Toggle Switch"

# ╔═╡ cf20c08b-5226-4c6c-8741-6857ede519a2
md"_Add required function to implement model_"

# ╔═╡ 1dc242f8-b0a1-11eb-2585-a121968b7921
begin
	mma(x) = 1/(1+x)
	mmr(x) = 1/(1+1/x)
	hfr(n) = x -> 1/(1 + x^n)
	hfa(n) = x -> 1/(1 + (1/x)^n)
	hs(λ, n) = x-> hfr(n)(x) + λ * hfa(n)(x)
	hs(λ, n, x₀) = x-> hs(λ, n)(x/x₀)
end

# ╔═╡ d7ef6de7-145c-4478-aa59-011562a5b8de
md"_Implement model piecewise (separate production and degradation terms) for simulations and computation_"

# ╔═╡ 812b4a91-8c80-4499-abd4-e65a464db31b
begin
	n = 4 # hill coefficient
	λ₁ = 3 # for bistable
	λ₂ = 10 # for tristable
	λₙ = 0.1 # repressor
	x₁ = 80 # cutoff count bistable
	x₂ = 25 # cutoff count tristable
	xₙ = 20 # cuttoff count repress
	p = 5 # unregulated production rate
	d = 0.1 # death rate
	proda(a,b) = p*hs(λ₂, n, x₂)(a)*hs(λₙ, n, xₙ)(b)
	dega(a, b) = d*a
	dadt(a, b) = proda(a, b) - dega(a, b) 
	prodb(a, b) = proda(b, a)
	degb(a, b)= d*b
	dbdt(a, b) = prodb(a,b) - degb(a, b)
	dfx = dadt
	dfy = dbdt
end

# ╔═╡ 4d905902-e730-4b79-a516-94ec9e236d59
begin
	fx(x::Vector) = dfx(x[1], x[2])
    fy(x::Vector) = dfy(x[1], x[2])
    f(x::Vector) = [fx(x), fy(x)]
end

# ╔═╡ 8416fdf7-1d4a-4af7-ba12-30f609c5dcdf
md"_Run Helmholtz decomposition using ForwardDiff for the divergence_"

# ╔═╡ 3ffd11ce-5388-4dfa-81ec-535c633ff378
begin
	# savefig(p_pot, "ch4_comp_pot.pdf")
	# savefig(p_rhs, "ch4_comp_rhs.pdf")
end

# ╔═╡ c105a211-66fb-495d-83bc-8ecf5679c37a
md"_Compute Vector Field_"

# ╔═╡ 006d5d3c-1b9e-442e-8fca-d7aa3a3abd35
begin
	n_min = 0
	n_max = 550
end

# ╔═╡ da1ae333-aafb-4ca7-a332-5d17a287c3d9
begin
	dq = (n_max - n_min) / 10.
	xs = n_min:dq:n_max+1
	ys = n_min:dq:n_max+1
	
	dxs = n_min:.5:n_max+1
	dys = n_min:.5:n_max+1

	df(x, y) = normalize([dfx(x, y), dfy(x,y)]) .* 30

	xxs = [x for x in xs for y in ys]
	yys = [y for x in xs for y in ys]

	quiver(xxs, yys, quiver=df)
	contour!(dxs, dys, dfx, levels=[0], color=:orange, label="X Nullcline", colorbar = false)
	contour!(dxs, dys, dfy, levels=[0], color=:green, label = "Y Nullcline", colorbar = false, xlabel=L"X", ylabel=L"Y", legend=:topright,)
end

# ╔═╡ ee260481-d1ae-4a38-b922-6e10bbfe945e
begin
	random_count() = floor(Int, n_min + rand() * (n_max - n_min))
	rand_u0() = [random_count(), random_count()]
	u0 = rand_u0()
	u0 = [5, 500]
	t = (0.0, 30000.0)
end

# ╔═╡ c077f2a7-6dd8-41a6-88ce-56ee38707081
md"_Stochastic DiffEq of Langevin equation for this system_

Note: `abs` is not supposed to be there but the simulation fails without it."

# ╔═╡ 32a77522-b389-4435-b92f-d05449f8b8d1
begin
	function f(du, u, p, t)
		du[1] = dfx(u...)
		du[2] = dfy(u...)
		du[1], du[2]
	end

	g(du, u, p, t) = begin
	  du[1,1] = sqrt(abs(proda(u[1], u[2])))
	  du[1,2] = -sqrt(abs(degb(u[1], u[2])))
	  du[1,3] = 0
	  du[1,4] = 0
	  du[2,1] = 0
	  du[2,2] = 0
	  du[2,3] = sqrt(abs(prodb(u[1], u[2])))
	  du[2,4] = -sqrt(abs(degb(u[1], u[2])))
	end
end

# ╔═╡ 1cd42539-b8db-4e2b-91a3-2038f299cf43
begin
	
	minval = 1
	maxval =  505
	nsample = 256
	rhs_xs = range(minval, maxval, length=nsample)
	rhs_ys = range(minval, maxval, length=nsample)

	divergence(F::Function, pt) = sum(
    	diag(
        	ForwardDiff.jacobian(F, pt)
        	)
    	)

	rhs_new = [divergence(f, [x, y]) for x=rhs_xs, y=rhs_ys]
end

# ╔═╡ bfb2c556-d3ec-449e-9f09-30d253bf3938
begin
	p_rhs = heatmap(rhs_xs, rhs_ys, rhs_new, xlabel=L"X", ylabel=L"Y", colorbar=:none)
end

# ╔═╡ 5989e134-f90b-43f0-bb2f-b2ffc5c74cb8
begin
	sde_prob_ssa= SDEProblem(f, g, u0, t, noise_rate_prototype=zeros(2,4))
	sde_sol = solve(sde_prob_ssa)
end

# ╔═╡ 46b15beb-6ee0-405e-96ca-6c0330a3c461
plot(sde_sol)

# ╔═╡ 1e7f2f9f-b8e6-44f8-8259-e2423368679d
md"_Gillespie Jump Process formulation for the same problem_"

# ╔═╡ f173fe40-ee7d-4513-8511-a18ec553ef5f
begin
	proda!(integrator) = integrator.u[1] += 1
	prodb!(integrator) = integrator.u[2] += 1
	dega!(integrator) = integrator.u[1] -= 1
	degb!(integrator) = integrator.u[2] -= 1
	jumps = Dict(proda => proda!, prodb => prodb!, dega => dega!, degb => degb!)
	constJumps = []
	for (rate, jump) in jumps
		push!(constJumps, ConstantRateJump((u, p, t) -> rate(u...), jump))
	end
	constJumps
end

# ╔═╡ 5c2c49e7-1948-41bb-82cc-f6a2460956fc
begin
	dprob_ssa = DiscreteProblem(u0, t)
	jprob_ssa = JumpProblem(dprob_ssa, Direct(), constJumps...)
end

# ╔═╡ 83b082e0-643e-4606-9db9-2957f3c6a470
ssa_sol = solve(jprob_ssa, SSAStepper())

# ╔═╡ 68c0fa0f-ff2d-4208-9a7d-415d0fa0abd1
stoch_tri = plot(ssa_sol, labels=[L"X" L"Y"], xlabel=L"t", legend=(0.9, 0.5))

# ╔═╡ 0715324c-8698-4a22-a146-9bc4c29c3a63
stoch_tri3 = plot(ssa_sol, vars=[(1, 2)], xlabel=L"X", ylabel=L"Y",legend=:none)

# ╔═╡ 73482204-dc5d-4fd5-aabb-72f284ca4416
pstoch = plot(stoch_tri, stoch_tri3, layout=(1, 2), size=(900, 500), title = ["A" "B"], titlelocation = :left)

# ╔═╡ 98664d78-abc5-45ca-a750-6184e0c285a8
# savefig(pstoch, "ch4_tri_stoch.pdf")

# ╔═╡ 2545abfb-df64-4a56-b217-d11fb7f2e95f
md"_Setup helper functions for running longer simulations_"

# ╔═╡ 0276796e-4f61-4d86-95b3-11df1b0ae242
function get_u0(i)
	offset = 1.0
	new_u0 = [0.0, 0.0]
	if i % 3 == 0
		new_u0 = [55, 55]
	elseif i % 3 == 1
		new_u0 .+= [5, 500]
	elseif i % 3 == 2
		new_u0 .+= [500, 5]
	end
	new_u0
end

# ╔═╡ a5d2a26e-5d33-480a-8712-8a38b43de1c9
md"_Storage for unnormalized probabilities that sum up the `dt`s in each bin_"

# ╔═╡ 31ed699a-77e2-4388-88a1-1dd740d56e40
md"_Rerun the following cell to add more values to the SSA simulation_"

# ╔═╡ 5250b601-7647-41cc-861a-393ad916cace
md"_Rerun the following cell to have the SDE simulation catch up to the SSA_"

# ╔═╡ d42b00d6-6f3f-44f9-8e5a-0ed8c8cc25ce
md"_the SDE probability distribution is complete non-sense_"

# ╔═╡ 076d3e3f-e3cd-4b60-8440-61c497b186a0
# savefig(ps, "ch4_ad_hoc.pdf")

# ╔═╡ a0bbd0f6-24d8-4705-b142-c1544346161c
md"## Appendix"

# ╔═╡ 125f4bfb-04b4-4fb3-8f44-413e8d8ee66c
md"_Helper function to generate -∇ ² to solve the Helmhotz equation_"

# ╔═╡ 13f7e53e-d2c9-49ec-b5c0-64e32299e758
begin
	eye(N, M) = sparse(I, N, M)
	eye(M) = eye(M, M)
	diff1(M) = [ [1.0 zeros(1, M - 1)]; diagm(1 => ones(M - 1)) - eye(M) ]
	sdiff1(M) = sparse(diff1(M))

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

# ╔═╡ bcc69011-ae43-452e-b5bd-5c5eecaaae19
begin
	_, _, Δ = laplacian2d(nsample, nsample, maxval - minval, maxval - minval)
	sol = reshape(Δ \ reshape(rhs_new, length(rhs_new)),  size(rhs_new)...)
end

# ╔═╡ fdac6046-4801-434a-9d03-01e4c2e9403c
p_pot = surface(rhs_xs, rhs_ys, sol, xlabel=L"X", ylabel=L"Y", colorbar=:none)

# ╔═╡ 60be1d36-ebfa-4ab7-ba68-5c35b6d41aa0
md"_Helper function to compute 2D histograms for Probability distributions_"

# ╔═╡ f245c60d-84f8-4f24-b2a0-c55abd991641
begin
	nbins = 701
	bin_min = 0
	bin_max = 700
	xbins = range(bin_min, bin_max, length=nbins)
	ybins = range(bin_min, bin_max, length=nbins)

	function bucket_idx(val, vec)
		idx = 1
		while val > vec[idx]
			idx += 1
		end
		idx
	end
	
	function althist(x, y, dt, xedges, yedges) 
	   counts = spzeros(length(yedges)-1, length(xedges)-1)
		for i=1:length(x) - 1
			r = bucket_idx(y[i], yedges)
			c = bucket_idx(x[i], xedges)
			counts[r,c] += dt[i]
		end
 		counts
	end
end

# ╔═╡ 18b7a92a-57f7-4b43-9d59-65b24889b676
begin
	unnorm_ssa = spzeros(length(xbins)-1, length(ybins)-1)
	unnorm_sde = spzeros(length(xbins)-1, length(ybins)-1)
	tspan = (0., 5e4)
end

# ╔═╡ 049020d7-1d6f-4603-9827-c1b1f22f7de6
prob_ssa = unnorm_ssa ./ sum(unnorm_ssa)

# ╔═╡ 6800f1bc-793e-49df-a60a-57305af8251a
prob_sde = unnorm_sde ./ sum(unnorm_sde)

# ╔═╡ f6a2e316-462e-4d3a-a6b2-37f02896599e
begin
	plotly()
	pl = surface(prob_sde, xlabel="X", ylabel="Y")
	gr()
	pl
end

# ╔═╡ 862c3e0a-261a-4e31-b986-f44d1056f66f
begin
	global n_points_sim
	for i in 1:3*10
		long_jprob_ssa = JumpProblem(
			DiscreteProblem(get_u0(i), tspan), Direct(), constJumps...)
		long_ssa_sol = solve(long_jprob_ssa, SSAStepper())
		unnorm_ssa .+= althist(long_ssa_sol[1,:], long_ssa_sol[2, :], diff(long_ssa_sol.t[:]), xbins, ybins)
	end
	unnorm_ssa
end

# ╔═╡ fe812819-697c-4840-a5df-f8ea1dd5e59b
begin
	i = 45 
	num_sde_run_failures = 44
	while sum(unnorm_sde) < sum(unnorm_ssa)
		global i
		global num_sde_run_failures

		
		sde_run = SDEProblem(f, g, get_u0(i), tspan, noise_rate_prototype=zeros(2,4))
		sol = solve(sde_prob_ssa)
	
		if sol.t[end] < tspan[end]
			# the simulation did not complete succesfully.
			# This is a sign of divergence and a failing approximation
			num_sde_run_failures += 1
		end
		unnorm_sde .+= althist(sol[1,:], sol[2, :], diff(sol.t[:]), xbins, ybins)
		i += 1
	end
	unnorm_sde, num_sde_run_failures, i
end

# ╔═╡ d9665278-3e2d-47b9-b955-1ddee1fe61e9
begin
	xplot = range(bin_min, bin_max, length=nbins-1)
	yplot = range(bin_min, bin_max, length=nbins-1)
end

# ╔═╡ a26b07ac-5b4d-40c0-b84c-7b86c19a5362
 begin
	pp1 = surface(xplot, yplot,-log.(prob_ssa), c = :thermal, title="ad hoc Quasi-potential")
	pp2 =  surface(xplot, yplot, prob_ssa, c = :grayC, title="Prob_ssaability Distribution", zaxis=(formatter=y->string(round(Int, y / 10^-4))*"e-4" ) )
	ps = plot(pp2, pp1, layout =(1, 2), size=(800, 600), xlabel=L"X", ylabel=L"Y", colorbar=:none)
end

# ╔═╡ Cell order:
# ╟─13137d52-2ede-4a63-8eac-43b13def7466
# ╠═193ee450-7b32-4394-9401-7b91bce21f31
# ╠═385b2bda-24d8-464c-9257-c4393a0b1c29
# ╟─cf20c08b-5226-4c6c-8741-6857ede519a2
# ╠═1dc242f8-b0a1-11eb-2585-a121968b7921
# ╟─d7ef6de7-145c-4478-aa59-011562a5b8de
# ╠═812b4a91-8c80-4499-abd4-e65a464db31b
# ╠═4d905902-e730-4b79-a516-94ec9e236d59
# ╟─8416fdf7-1d4a-4af7-ba12-30f609c5dcdf
# ╠═1cd42539-b8db-4e2b-91a3-2038f299cf43
# ╠═bfb2c556-d3ec-449e-9f09-30d253bf3938
# ╠═bcc69011-ae43-452e-b5bd-5c5eecaaae19
# ╠═fdac6046-4801-434a-9d03-01e4c2e9403c
# ╠═3ffd11ce-5388-4dfa-81ec-535c633ff378
# ╟─c105a211-66fb-495d-83bc-8ecf5679c37a
# ╠═006d5d3c-1b9e-442e-8fca-d7aa3a3abd35
# ╠═da1ae333-aafb-4ca7-a332-5d17a287c3d9
# ╠═ee260481-d1ae-4a38-b922-6e10bbfe945e
# ╟─c077f2a7-6dd8-41a6-88ce-56ee38707081
# ╠═32a77522-b389-4435-b92f-d05449f8b8d1
# ╠═5989e134-f90b-43f0-bb2f-b2ffc5c74cb8
# ╠═46b15beb-6ee0-405e-96ca-6c0330a3c461
# ╟─1e7f2f9f-b8e6-44f8-8259-e2423368679d
# ╠═f173fe40-ee7d-4513-8511-a18ec553ef5f
# ╠═5c2c49e7-1948-41bb-82cc-f6a2460956fc
# ╠═83b082e0-643e-4606-9db9-2957f3c6a470
# ╠═68c0fa0f-ff2d-4208-9a7d-415d0fa0abd1
# ╠═0715324c-8698-4a22-a146-9bc4c29c3a63
# ╠═73482204-dc5d-4fd5-aabb-72f284ca4416
# ╠═98664d78-abc5-45ca-a750-6184e0c285a8
# ╟─2545abfb-df64-4a56-b217-d11fb7f2e95f
# ╠═0276796e-4f61-4d86-95b3-11df1b0ae242
# ╟─a5d2a26e-5d33-480a-8712-8a38b43de1c9
# ╠═18b7a92a-57f7-4b43-9d59-65b24889b676
# ╟─31ed699a-77e2-4388-88a1-1dd740d56e40
# ╠═862c3e0a-261a-4e31-b986-f44d1056f66f
# ╟─5250b601-7647-41cc-861a-393ad916cace
# ╠═fe812819-697c-4840-a5df-f8ea1dd5e59b
# ╠═049020d7-1d6f-4603-9827-c1b1f22f7de6
# ╠═6800f1bc-793e-49df-a60a-57305af8251a
# ╟─d42b00d6-6f3f-44f9-8e5a-0ed8c8cc25ce
# ╠═f6a2e316-462e-4d3a-a6b2-37f02896599e
# ╠═d9665278-3e2d-47b9-b955-1ddee1fe61e9
# ╠═a26b07ac-5b4d-40c0-b84c-7b86c19a5362
# ╠═076d3e3f-e3cd-4b60-8440-61c497b186a0
# ╟─a0bbd0f6-24d8-4705-b142-c1544346161c
# ╟─125f4bfb-04b4-4fb3-8f44-413e8d8ee66c
# ╠═13f7e53e-d2c9-49ec-b5c0-64e32299e758
# ╟─60be1d36-ebfa-4ab7-ba68-5c35b6d41aa0
# ╠═f245c60d-84f8-4f24-b2a0-c55abd991641
