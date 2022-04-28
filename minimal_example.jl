cd(@__DIR__)
using Pkg
Pkg.activate(".")
##

include("sp_model.jl")
using DataFrames
using CSV

Random.seed!(1)

##
# Define the data we are working on
timesteps = 1:160

pv = CSV.read("timeseries/basic_example_normalized.csv", DataFrame)[timesteps, 3]
wind = CSV.read("timeseries/basic_example_normalized.csv", DataFrame)[timesteps, 4]
demand = CSV.read("timeseries/basic_example_normalized.csv", DataFrame)[timesteps, 2]

##

pars = copy(default_es_pars)

##

using Statistics
average_hourly_demand = mean(demand)

pars[:recovery_time] = 24
pars[:c_storage] = 100.
pars[:c_pv] = 300.
pars[:c_wind] = 800.
pars[:c_sto_op] = 0.00001

heatdemand = copy(demand)./300.
heatdemand0 = zeros(length(demand))
es = define_energy_system(pv, wind, demand, heatdemand0; p = pars, strict_flex = true)

##
n = 100
F_max = average_hourly_demand * 0.1 # Have ancillary services equal to 10% of our typical demand
t_max = length(pv) - es.parameters[2].defaults[:recovery_time]
scens = simple_flex_sampler(n, F_max, t_max)

##

using Cbc

sp = instantiate(es, scens, optimizer = Cbc.Optimizer)

##

optimize!(sp)

##

od = optimal_decision(sp)

ov = objective_value(sp)
# We save objective value and result of evaluate_decision for each scenario
ovs = [objective_value(sp, i) for i in 1:length(scens)] # Orde of magnitude wise this should probably be h(x, xi)
eds = [evaluate_decision(sp, od, scen) for scen in scens]

ed0 = evaluate_decision(sp, od, no_flex_pseudo_sampler()[1]) 

##
@show ov - evaluate_decision(sp, od) ; # 0.0

nothing
##
@show ed0 # -181827.20023213763
@show mean(eds) .- ov # -73666.1499255957
@show mean(ovs) .- ov # 140615.50919693563

nothing
##
# If f() = evaluate_decision(sp, od, s)-objective_value(sp,2,s), then eds[i]-ovs[i] should be scenario independent, that is constant. We check if that's true:
@show maximum(eds .- ovs) # -181827.20023213763
@show minimum(eds .- ovs) #  -227655.07249998135
# Unfortunately it's not the same.

nothing
##
# We also found a bug in cache_solution. After it objective_value(sp,2,i) == objective_value(sp) for any i.
# cache_solution!(sp)

##

# ovs2 = [objective_value(sp, i) for i in 1:length(scens)]

# unique(ovs2 .- ov) # [0.0]