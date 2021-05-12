### A Pluto.jl notebook ###
# v0.14.5

using Markdown
using InteractiveUtils

# ╔═╡ 14a4cc28-ab71-11eb-0021-1506fbc8ae2f
using Plots, LaTeXStrings

# ╔═╡ 639c60aa-be9e-4973-849d-345a61234663
md"# Chapter 1 Figures for rates"

# ╔═╡ d47fe2e9-07a1-492a-a6cd-e4a9f92cb5aa
begin
	mma(x) = 1/(1+x)
	mmr(x) = 1/(1+1/x)
	hfr(n) = x -> 1/(1 + x^n)
	hfa(n) = x -> 1/(1 + (1/x)^n)
	hs(λ, n) = x-> hfr(n)(x) + λ * hfa(n)(x)
end

# ╔═╡ 260fcd7d-ebf4-4859-8121-6455ff1e3c81
x = 0:0.01:2.0

# ╔═╡ 0dd7b136-8ee0-4d60-b8e5-c12094c5ac71
m2 = plot(mma, x, legend=:none, ylimits = (0. ,1.), xformatter=_->"", yformatter=_->"", title="Michaelis-Menten Repressor", yticks=[0, 0.5, 1.0])

# ╔═╡ c08d7b09-a6e4-4107-a9a8-54484f4f5829
m1 = plot(mmr, x, legend=:none, ylimits = (0. ,1.),
	xformatter=_->"",
	title="Michaelis-Menten Activator", ylabel="Normalized Promoter Activity", yticks=[0, 0.5, 1.0])

# ╔═╡ 47b79606-b710-4928-88e3-902dda002560
begin
	plot()
	for n = 2:15
		plot!(hfa(n), x, label=L"n = %$n")
	end
	m3 = plot!(legend=:none, xlabel=L"\frac{S_X}{K_X}", ylimits = (0. ,1.), title="Hill Activator", ylabel="Normalized Promoter Activity", yticks=[0, 0.5, 1.0])
end

# ╔═╡ 28d6f3cd-af64-43e0-b951-a104955e89b2
begin
	plot()
	for n = 2:15
		plot!(hfr(n), x)
	end
	m4 = plot!(legend=:none, xlabel=L"\frac{S_X}{K_X}", ylimits = (0. ,1.),
		yformatter=_->"",
		title="Hill Repressor", yticks=[0, 0.5, 1.0])
end

# ╔═╡ 3b79d3b7-7a58-4b88-82cf-507776972aca
begin
	plot()
	plot!(hs(.5, 3), x, label=L"\lambda = 0.5")
	plot!(hs(1, 3), x, label=L"\lambda = 1")
	plot!(hs(2, 3), x, label=L"\lambda = 2")
	shil = plot!(xlabel=L"\frac{S_X}{K_X}",ylabel="Normalized Promoter Activity",title="Shifted Hill", yticks=[0, 0.5, 1.0, 1.5, 2.0])
	savefig(shil, "ch1_shifted.pdf")
	shil
end

# ╔═╡ 573d16f2-b276-4a34-9ec1-c12e3b59af9f
begin
	figactivity = plot(m1,m2,m3,m4, layout=4, size=(750,750))
	figactivity
end

# ╔═╡ a7ef3696-ff92-4264-9dcc-4a38f21d9852
savefig(figactivity, "ch1_activity.pdf")

# ╔═╡ Cell order:
# ╠═14a4cc28-ab71-11eb-0021-1506fbc8ae2f
# ╟─639c60aa-be9e-4973-849d-345a61234663
# ╠═d47fe2e9-07a1-492a-a6cd-e4a9f92cb5aa
# ╠═260fcd7d-ebf4-4859-8121-6455ff1e3c81
# ╠═0dd7b136-8ee0-4d60-b8e5-c12094c5ac71
# ╠═c08d7b09-a6e4-4107-a9a8-54484f4f5829
# ╠═47b79606-b710-4928-88e3-902dda002560
# ╠═28d6f3cd-af64-43e0-b951-a104955e89b2
# ╠═3b79d3b7-7a58-4b88-82cf-507776972aca
# ╠═573d16f2-b276-4a34-9ec1-c12e3b59af9f
# ╠═a7ef3696-ff92-4264-9dcc-4a38f21d9852
