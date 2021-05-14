### A Pluto.jl notebook ###
# v0.14.5

using Markdown
using InteractiveUtils

# ╔═╡ aab4db99-8eaf-4b70-add7-d470e6b6b6e2
using Plots, LaTeXStrings, DifferentialEquations, Distributions

# ╔═╡ 460194e8-add2-11eb-0f92-27b8fffb5046
md"# Chapter 1 Figures. Statistics"

# ╔═╡ de27faec-4e2b-4520-9d3f-1ac15344bab1
random_count() = floor(Int, n_min + rand() * n_max)

# ╔═╡ a8fa0a9a-4fb7-4380-87dd-2534687409c4
begin
	affect1!(integrator) = integrator.u += 1
	affect2!(integrator) = integrator.u -= 1
	prod1Jump(A) = ConstantRateJump((u, p, t) -> A, affect1!)
	deg1Jump = ConstantRateJump((u, p, t) -> u, affect2!)
	u01 = 0
	t1 = (0.0, 15.0)
	dprob1 = DiscreteProblem(u01, t1)
	jprob1 = JumpProblem(dprob1, Direct(), prod1Jump(100), deg1Jump)
	continous(A) = t -> -A*ℯ^(-t) + A
end

# ╔═╡ 343e56a2-4837-49a6-aa69-5a285d362c03
sol1 = solve(jprob1, SSAStepper())

# ╔═╡ 238a13fa-6d94-448d-a236-fdde5b78ad98
begin
	plot(sol1, label="Stochastic", legend=:topleft)
	p1 = plot!(continous(100), sol1.t, label="Continuous", xlabel=L"t", ylabel=L"n", ylimit=(0,145))
end

# ╔═╡ 67a22053-294a-47d1-9793-2188398de91c
begin
	last_ones = []
	num_samples=10000 
	for _ ∈ 1:num_samples
		push!(last_ones, solve(jprob1, SSAStepper()).u[end])
	end
	last_ones
end

# ╔═╡ 67f8f4d0-0e6d-4857-9ac6-ea3b9f1eca3f
begin
	# poisson(mean)
	histogram(last_ones, label="Distribution of final values", color=:orange, orientation=:h)
	p2 = plot!(rand(Poisson(100), num_samples), seriestype=:stephist, label="Sampled Poisson Distribution", color=:blue, orientation=:h, ylimit=(0,145), ticks=nothing, yaxis=false, xaxis=false)
end

# ╔═╡ 6c9d9245-317a-47ca-83c4-b00523eaf3fd
pf = plot(p1, p2)

# ╔═╡ 2be2984e-1315-41a5-b7c4-63e79fad18f2
savefig(pf, "ch1_stochastic.pdf")

# ╔═╡ Cell order:
# ╠═aab4db99-8eaf-4b70-add7-d470e6b6b6e2
# ╟─460194e8-add2-11eb-0f92-27b8fffb5046
# ╠═de27faec-4e2b-4520-9d3f-1ac15344bab1
# ╠═a8fa0a9a-4fb7-4380-87dd-2534687409c4
# ╠═343e56a2-4837-49a6-aa69-5a285d362c03
# ╠═238a13fa-6d94-448d-a236-fdde5b78ad98
# ╠═67a22053-294a-47d1-9793-2188398de91c
# ╠═67f8f4d0-0e6d-4857-9ac6-ea3b9f1eca3f
# ╠═6c9d9245-317a-47ca-83c4-b00523eaf3fd
# ╠═2be2984e-1315-41a5-b7c4-63e79fad18f2
