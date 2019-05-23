# developed with Julia 1.1.1
#
# offline step for Stochastic Dynamic Programming 


function admissible_state(x::Array{Float64}, states::Grid)
	"""check if x is in states: return a boolean"""

	for i in 1:length(x)

		if x[i] < states[i][1]
			return false
		elseif x[i] > states[i][end]
			return false
		end

	end

	return true

end