using DataStructures # for DefaultDict

# Data type aliases
const State = Observation = Action = Trial = Int64
const Reward = Time = Temp = Float64
const Actions = Vector{Action}
const Values = DefaultDict{Tuple{State, Action}, Reward}
const Counterₛₜ = DefaultDict{Tuple{State, Time}, Int64}
const Counterₛₐₜ = DefaultDict{Tuple{State, Action, Time}, Int64}

actions = Actions([1,2]) # List of actions
memory = BitVector([0]) # One-bit memory setting
push!(actions, 0) # Append bit-setting action

# Experience sequence
mutable struct Sequence
	s::State
	a::Action
	r::Reward
	t::Time
	Sequence(s,a,t) = new(s, a, R(s,a), t)
	Sequence(s,a,r,t) = new(s, a, r, t)
end

const cmax = 1.0 # Maximum temperature
const cmin = 0.2 # Minimum temperature

Q = Values(0) # Q-value look-up table
Nₛₐₜ = Counterₛₐₜ(0) # Counter for (s,a,t)
Nₛₜ  = Counterₛₜ(0) # Counter for (s,t)

α(N::Trial; α0=0.5) = α0 + 1/(10N) # Learning rate with decay
δc(N::Trial) = (cmin/cmax)^(1/(N-1)) # Temperature decay
R(s::State, a::Action) = s == State(9) ? 1 : 0 # Reward function

# Increment the visit counters
function visit!(s::State, a::Action, t::Time)
	Nₛₐₜ[s,a,t] += 1
	Nₛₜ[s,t] += 1
end

# Main core of VAPS(1)

e(z::Sequence; b=0, γ=0.9) = b - γ^z.t * z.r

function boltzmann_distribution(Q::Values, s::State, a::Action, c::Temp)
	return exp(Q[s,a]/c) / sum(a′->exp(Q[s,a′]/c), actions)
end

function exploration_trace(Q::Values, s::State, a::Action, t::Time, N::Trial)
	c::Temp = max(cmax - δc(N), cmin)
	return 1/c * (Nₛₐₜ[s,a,t] - Nₛₜ[s,t]*boltzmann_distribution(Q, s, a, c))
end

function update_q!(Q::Values, z::Sequence, N::Trial)
	(s::State, a::Action, t::Time) = (z.s, z.a, z.t)
	visit!(s, a, t)
	Q[s,a] = Q[s,a] - α(N)*e(z)*exploration_trace(Q, s, a, t, N)
end


# Not in writeup
ε(t) = sum(z->e(z), Z[1:t])
Z = [Sequence(1,1,0.0,1.0), Sequence(2,1,0.0,2.0), Sequence(3,1,1.0,3.0)]

# function VAPS1(Q::Values, s::State) # TODO: Put it all together in a test run.