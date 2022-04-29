using StochasticPrograms
using Random
using Statistics

Random.seed!(2)

##
n = 5
t_max = 10
prices = collect(1:t_max)

scens = [@scenario t_r = rand(1:t_max-2) probability = 1/n for i in 1:n]

##

system = @stochastic_model begin 
    @stage 1 begin
        @decision(model, 3. >= a[t in 1:t_max] >= 0.)
        @constraint(model, sum(a) == 20.)
        @objective(model, Min, sum(a .* prices))
    end
    @stage 2 begin
        @uncertain t_r
        trs = t_r:t_r+2
        @known(model, a)
        @recourse(model, a2[t in trs] >= 0.)
        @constraint(model, sum(a2) == sum(a[trs]))
        @constraint(model, a2[t_r + 1] == 0)
        @objective(model, Min, sum((a2[trs] .- a[trs]) .* prices[trs]))
    end
end

##


using Cbc

sp = instantiate(system, scens, optimizer = Cbc.Optimizer)

##

optimize!(sp)

##

od = optimal_decision(sp)

ov = objective_value(sp)

ovs = [objective_value(sp, i) for i in 1:length(scens)]
eds = [evaluate_decision(sp, od, scen) for scen in scens]

##
@show ov - evaluate_decision(sp, od) ; # 0.0

nothing
##
# Neither eds nor ovs have ov as the average:

@show mean(eds) .- ov # 24.6
@show mean(ovs) .- ov # -77.0

nothing
##
# If f() = evaluate_decision(sp, od, s)-objective_value(sp,2,s), then eds[i]-ovs[i] should be scenario independent:

@show maximum(eds .- ovs) # 109.0
@show minimum(eds .- ovs) # 91.0

nothing
