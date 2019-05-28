# developed with Julia 1.1.1
#
# offline step for Stochastic Dynamic Programming 


function admissible_state!(x::Array{Float64}, states::Grid)
	"""check if x is in states: project x in states and return a boolean"""

	is_admissible = true

	for i in 1:length(x)

		if x[i] < states[i][1]
			is_admissible = false
			x[i] = states[i][1]
		elseif x[i] > states[i][end]
			is_admissible = false
			x[i] = states[i][end]
		end

	end

	return is_admissible

end