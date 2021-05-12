### A Pluto.jl notebook ###
# v0.14.5

using Markdown
using InteractiveUtils

# ╔═╡ ad96c6b4-add2-11eb-1a72-8993033e1259
using Plots, LaTeXStrings, QuadGK, Roots, DifferentialEquations

# ╔═╡ 5168971b-2b60-494c-a8d8-cdfa8e1b8102
md"# Chapter 3 figures. SImple Example"

# ╔═╡ 2a1d5060-d991-460d-9152-520940a95cb1
begin
	ys=0.1:0.01:3.0
	ha = x -> 1/(1 + (1/x)^3) + 0.31
	ddd = x-> .51x + 0.21
	plot(ha, ys, ylimits=(0, 2))
	plot!(ddd, ys)
end

# ╔═╡ 02d0fcf2-ca87-49ae-a8f3-f14310d0182a
begin
	n_max = 200.
	n_min = 1 # need to avoid 0 for log we'll evaluate later
	n = n_min:0.1:n_max
	γ = 1
	ks = 1e-4
	ν0 = 12.5
	ν1 = 200
	production(n) = (ν0+ ν1 * ks*n^2)/(1+ks*n^2)
	degradation(n) = γ*n
end

# ╔═╡ 13ddc727-e9bd-44f2-ba5e-c6c311b41723
begin
	
	plot(production, n, label=L"f(n)")
	p1 = plot!(degradation, n, label=L"g(n)", xlabel=L"n", title="Rate of Change", scale=:log10, xlimits=(15, n_max), ylimits=(15, 200), legend=:topleft)
# plot!(n->production(n)  - degradation(n), n, label=L"\frac{dn}{dt}", 
# 		ylims=(-20, 150), xlims=(0, 150), legend=:topleft, title="Time evolution")
end

# ╔═╡ 348718a0-8e19-4911-8998-5de9689bce1d
fixed_points = find_zeros(n-> production(n) - degradation(n), n_min, n_max)

# ╔═╡ acb0adcf-fc19-45cf-bf0f-eb5532743b1a
begin
	f = production
	g = degradation
	
	ϕ(n) = quadgk(n -> -2*(f(n) -g(n))/(f(n) + g(n)), n_min, n)[1]
end

# ╔═╡ b4c511ea-fc96-475e-a5fc-1c9e282ca0e5
ϕ_vec = ϕ.(n)

# ╔═╡ 3373f26d-a6a4-4378-a8e1-c011c2e9f7a9
p2 = plot(n, ϕ_vec, ylabel=L"\phi(n)", xlabel=L"n", label="", title="Quasi-potential")

# ╔═╡ 178564ff-da53-4dd5-ba1f-b3e5db24b2ea
begin
	plot(n->log10.(f(n) + g(n)), n, label=L"f + g")
	plot!(n, log10.(ℯ.^(-ϕ_vec)), label=L"e^{-\phi}")
	p3 = plot!(xaxis=:log10, legend=:topleft)
end

# ╔═╡ a7e9be12-a1d4-4f1a-bbfc-d2642e5a9dd5
begin
	fn = f.(n)
	gn = g.(n)
	p_unnormal(n) =  ℯ^(-ϕ(n)) / (f(n)+g(n))
	norm = quadgk(p_unnormal, n_min, n_max)[1]
	p_normal(n) = 1/norm * p_unnormal(n)
	p4 = plot(p_normal,n,  title = "Normalized Probability", label="")
end

# ╔═╡ 50c911b1-57cf-4de4-9d65-054318c445fe
begin
	pf = plot(p1, p2, p4,p5, layout = grid(2,2), size=(750, 750))
	# savefig(pf, "ch3_bistable_prob.pdf")
	pf
end

# ╔═╡ f5b8a081-222f-4dd6-ae4c-af7060aa4223


# ╔═╡ 467386e3-2ee8-46d4-9b40-831ae2778a57
random_count() = floor(Int, n_min + rand() * n_max)

# ╔═╡ 6ae4ed84-80c2-4c06-b69a-c81e6485b78b
begin
	affect1!(integrator) = integrator.u += 1
	affect2!(integrator) = integrator.u -= 1
	prodJump = ConstantRateJump((u, p, t) -> f(u), affect1!)
	degJump = ConstantRateJump((u, p, t) -> g(u), affect2!)
	u0 = random_count()
	t = (0.0, 5000.0)
	dprob = DiscreteProblem(u0, t)
	jprob = JumpProblem(dprob, Direct(), prodJump, degJump)
end

# ╔═╡ f32b2ab8-a941-463b-bb27-968c09e24777
begin
	sim_res = solve(jprob, SSAStepper())
	# p5 = plot(sim_res, xlabel=L"t", label=L"n(t)", title="Simulation trace")
end

# ╔═╡ 456090be-e1c0-452e-b0ad-98fbc9f0576f
function calculate_time_fraction(sim_result, split)
	dts = diff(sim_result.t)
	dt_vals = collect(zip(dts, map(last, sim_result.u[1:end])))
	high = sum(map(first, filter(dtu -> last(dtu) > split, dt_vals)))
	low = sum(map(first, filter(dtu -> last(dtu) < split, dt_vals)))
	high, low
end

# ╔═╡ 658d3346-b602-4ddb-9905-55cbbf7c5417
begin
	times = calculate_time_fraction(sim_res, fixed_points[2])
	times ./ sum(times)
end

# ╔═╡ 23494ad1-82e3-4eee-86e3-695511de502e
function big_sim(num_sim = 25)
	function prob_func(prob, i, repeat)
        remake(prob, u0=random_count())
    end
	num_steps = 0
    ensemble_prob = EnsembleProblem(jprob, prob_func=prob_func)
	sim = solve(ensemble_prob, SSAStepper(), trajectories=num_sim)
	totals = [0., 0.]
	for j in 1:num_sim
		totals.+= calculate_time_fraction(sim[j], fixed_points[2])
		num_steps += length(sim[j])
	end
	totals, num_steps
end

# ╔═╡ 0e38a5c3-d970-4844-b560-fe4ed5452d85
totals, num_steps = big_sim()

# ╔═╡ 573a4914-b37e-4c2b-bd23-c10b58a6d4db
totals./sum(totals)

# ╔═╡ 86ee8326-51e1-453d-8d83-fd7021613c76
begin
	p_low = quadgk(p_normal, n_min, fixed_points[2])[1]
	p_high = quadgk(p_normal, fixed_points[2], n_max)[1]
	p_high, p_low
end

# ╔═╡ 4eed64ba-66e0-4024-8f26-39a4cb233191
u0, f(u0), g(u0)

# ╔═╡ Cell order:
# ╠═ad96c6b4-add2-11eb-1a72-8993033e1259
# ╟─5168971b-2b60-494c-a8d8-cdfa8e1b8102
# ╠═2a1d5060-d991-460d-9152-520940a95cb1
# ╠═02d0fcf2-ca87-49ae-a8f3-f14310d0182a
# ╠═13ddc727-e9bd-44f2-ba5e-c6c311b41723
# ╠═348718a0-8e19-4911-8998-5de9689bce1d
# ╠═acb0adcf-fc19-45cf-bf0f-eb5532743b1a
# ╠═b4c511ea-fc96-475e-a5fc-1c9e282ca0e5
# ╠═3373f26d-a6a4-4378-a8e1-c011c2e9f7a9
# ╠═178564ff-da53-4dd5-ba1f-b3e5db24b2ea
# ╠═a7e9be12-a1d4-4f1a-bbfc-d2642e5a9dd5
# ╠═50c911b1-57cf-4de4-9d65-054318c445fe
# ╠═f5b8a081-222f-4dd6-ae4c-af7060aa4223
# ╠═467386e3-2ee8-46d4-9b40-831ae2778a57
# ╠═6ae4ed84-80c2-4c06-b69a-c81e6485b78b
# ╠═f32b2ab8-a941-463b-bb27-968c09e24777
# ╠═456090be-e1c0-452e-b0ad-98fbc9f0576f
# ╠═658d3346-b602-4ddb-9905-55cbbf7c5417
# ╠═23494ad1-82e3-4eee-86e3-695511de502e
# ╠═0e38a5c3-d970-4844-b560-fe4ed5452d85
# ╠═573a4914-b37e-4c2b-bd23-c10b58a6d4db
# ╠═86ee8326-51e1-453d-8d83-fd7021613c76
# ╠═4eed64ba-66e0-4024-8f26-39a4cb233191
