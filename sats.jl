### A Pluto.jl notebook ###
# v0.14.5

using Markdown
using InteractiveUtils

# ╔═╡ 193ee450-7b32-4394-9401-7b91bce21f31
using LaTeXStrings, DifferentialEquations, LinearAlgebra, Plots, SparseArrays

# ╔═╡ 385b2bda-24d8-464c-9257-c4393a0b1c29
using ForwardDiff

# ╔═╡ 1dc242f8-b0a1-11eb-2585-a121968b7921
begin
	mma(x) = 1/(1+x)
	mmr(x) = 1/(1+1/x)
	hfr(n) = x -> 1/(1 + x^n)
	hfa(n) = x -> 1/(1 + (1/x)^n)
	hs(λ, n) = x-> hfr(n)(x) + λ * hfa(n)(x)
	hs(λ, n, x₀) = x-> hs(λ, n)(x/x₀)
end

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

# ╔═╡ 1cd42539-b8db-4e2b-91a3-2038f299cf43
begin
	
	minval = 1
	maxval =  505
	nsample = 1024
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

# ╔═╡ cd77f2c5-bc07-4a97-b576-664895e14814


# ╔═╡ ee260481-d1ae-4a38-b922-6e10bbfe945e
begin
	random_count() = floor(Int, n_min + rand() * (n_max - n_min))
	rand_u0() = [random_count(), random_count()]
	u0 = rand_u0()
	u0 = [5, 500]
end

# ╔═╡ 32a77522-b389-4435-b92f-d05449f8b8d1
# begin
# 	function f(du, u, p, t)
# 		du[1] = dfx(u...)
# 		du[2] = dfy(u...)
# 		du[1], du[2]
# 	end

# 	g(du, u, p, t) = begin
# 	  du[1] = 1.0
# 	  du[2] = 1.0
# 	end
# 	sde_prob= SDEProblem(f, g, u0, (0.0, 1000.0))
# 	sde_sol = solve(sde_prob)
# end

# ╔═╡ 46b15beb-6ee0-405e-96ca-6c0330a3c461
# plot(sde_sol)

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
	t = (0.0, 30000.0)
	dprob = DiscreteProblem(u0, t)
	jprob = JumpProblem(dprob, Direct(), constJumps...)
end

# ╔═╡ 83b082e0-643e-4606-9db9-2957f3c6a470
ssa_sol = solve(jprob, SSAStepper())

# ╔═╡ 68c0fa0f-ff2d-4208-9a7d-415d0fa0abd1
stoch_tri = plot(ssa_sol, labels=[L"X" L"Y"], xlabel=L"t", legend=(0.9, 0.5))

# ╔═╡ 0715324c-8698-4a22-a146-9bc4c29c3a63
stoch_tri3 = plot(ssa_sol, vars=[(1, 2)], xlabel=L"X", ylabel=L"Y",legend=:none)

# ╔═╡ 73482204-dc5d-4fd5-aabb-72f284ca4416
pstoch = plot(stoch_tri, stoch_tri3, layout=(1, 2), size=(900, 500), title = ["A" "B"], titlelocation = :left)

# ╔═╡ 98664d78-abc5-45ca-a750-6184e0c285a8
# savefig(pstoch, "ch4_tri_stoch.pdf")

# ╔═╡ 557499ef-0dd7-4539-8f2e-5a7f8385a4d8


# ╔═╡ 0276796e-4f61-4d86-95b3-11df1b0ae242
function get_u0(i)
	offset = 1.0
	new_u0 = [0.0, 0.0]
	if i % 4 == 0
		new_u0 = [55, 55]
	elseif i % 4 == 1
		new_u0 .+= [5, 500]
	elseif i % 4 == 2
		new_u0 .+= [500, 5]
	end
	new_u0
end

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
unnorm = spzeros(length(xbins)-1, length(ybins)-1)

# ╔═╡ 862c3e0a-261a-4e31-b986-f44d1056f66f
begin
	for i in 1:3*10
		u00 = get_u0(i)
		long_sim_prob = DiscreteProblem(u00, (0., 1e4))
		long_jprob = JumpProblem(long_sim_prob, Direct(), constJumps...)
		long_ssa_sol = solve(long_jprob, SSAStepper())
		unnorm .+= althist(long_ssa_sol[1,:], long_ssa_sol[2, :], diff(long_ssa_sol.t[:]), xbins, ybins)
	end
	unnorm
end

# ╔═╡ 049020d7-1d6f-4603-9827-c1b1f22f7de6
prob = unnorm ./ sum(unnorm)

# ╔═╡ 6800f1bc-793e-49df-a60a-57305af8251a


# ╔═╡ f6a2e316-462e-4d3a-a6b2-37f02896599e
# begin
# 	plotly()
# 	pl = surface(- log.(prob))
# 	gr()
# 	pl
# end

# ╔═╡ d9665278-3e2d-47b9-b955-1ddee1fe61e9
begin
	xplot = range(bin_min, bin_max, length=nbins-1)
	yplot = range(bin_min, bin_max, length=nbins-1)
end

# ╔═╡ 076d3e3f-e3cd-4b60-8440-61c497b186a0
# savefig(ps, "ch4_ad_hoc.pdf")

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

# ╔═╡ 3ffd11ce-5388-4dfa-81ec-535c633ff378
begin
	savefig(p_pot, "ch4_comp_pot.pdf")
	savefig(p_rhs, "ch4_comp_rhs.pdf")
end

# ╔═╡ 321a6ecf-5017-4f54-a497-7207b06849e5


# ╔═╡ dbf59c22-edd7-4ed2-8542-3467064bb512
	ps = plot(p_pot, p_rhs, layout =(1, 2), size=(800, 600), xlabel=L"X", ylabel=L"Y", colorbar=:none)

# ╔═╡ a26b07ac-5b4d-40c0-b84c-7b86c19a5362
 begin
	pp1 = surface(xplot, yplot,-log.(prob), c = :thermal, title="ad hoc Quasi-potential")
	pp2 =  surface(xplot, yplot, prob, c = :grayC, title="Probability Distribution", zaxis=(formatter=y->string(round(Int, y / 10^-4))*"e-4" ) )
	ps = plot(pp2, pp1, layout =(1, 2), size=(800, 600), xlabel=L"X", ylabel=L"Y", colorbar=:none)
end

# ╔═╡ Cell order:
# ╠═193ee450-7b32-4394-9401-7b91bce21f31
# ╠═385b2bda-24d8-464c-9257-c4393a0b1c29
# ╠═1dc242f8-b0a1-11eb-2585-a121968b7921
# ╠═812b4a91-8c80-4499-abd4-e65a464db31b
# ╠═4d905902-e730-4b79-a516-94ec9e236d59
# ╠═1cd42539-b8db-4e2b-91a3-2038f299cf43
# ╠═bfb2c556-d3ec-449e-9f09-30d253bf3938
# ╠═bcc69011-ae43-452e-b5bd-5c5eecaaae19
# ╠═fdac6046-4801-434a-9d03-01e4c2e9403c
# ╠═dbf59c22-edd7-4ed2-8542-3467064bb512
# ╠═3ffd11ce-5388-4dfa-81ec-535c633ff378
# ╠═006d5d3c-1b9e-442e-8fca-d7aa3a3abd35
# ╠═da1ae333-aafb-4ca7-a332-5d17a287c3d9
# ╠═cd77f2c5-bc07-4a97-b576-664895e14814
# ╠═ee260481-d1ae-4a38-b922-6e10bbfe945e
# ╠═32a77522-b389-4435-b92f-d05449f8b8d1
# ╠═46b15beb-6ee0-405e-96ca-6c0330a3c461
# ╠═f173fe40-ee7d-4513-8511-a18ec553ef5f
# ╠═5c2c49e7-1948-41bb-82cc-f6a2460956fc
# ╠═83b082e0-643e-4606-9db9-2957f3c6a470
# ╠═68c0fa0f-ff2d-4208-9a7d-415d0fa0abd1
# ╠═0715324c-8698-4a22-a146-9bc4c29c3a63
# ╠═73482204-dc5d-4fd5-aabb-72f284ca4416
# ╠═98664d78-abc5-45ca-a750-6184e0c285a8
# ╠═557499ef-0dd7-4539-8f2e-5a7f8385a4d8
# ╠═0276796e-4f61-4d86-95b3-11df1b0ae242
# ╠═18b7a92a-57f7-4b43-9d59-65b24889b676
# ╠═862c3e0a-261a-4e31-b986-f44d1056f66f
# ╠═f245c60d-84f8-4f24-b2a0-c55abd991641
# ╠═049020d7-1d6f-4603-9827-c1b1f22f7de6
# ╠═6800f1bc-793e-49df-a60a-57305af8251a
# ╠═f6a2e316-462e-4d3a-a6b2-37f02896599e
# ╠═d9665278-3e2d-47b9-b955-1ddee1fe61e9
# ╠═a26b07ac-5b4d-40c0-b84c-7b86c19a5362
# ╠═076d3e3f-e3cd-4b60-8440-61c497b186a0
# ╠═13f7e53e-d2c9-49ec-b5c0-64e32299e758
# ╠═321a6ecf-5017-4f54-a497-7207b06849e5
