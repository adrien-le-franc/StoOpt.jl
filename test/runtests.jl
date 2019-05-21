# developed with Julia 1.1.1
#
# tests for StoOpt package

using StoOpt, CSV, Test

current_directory = @__DIR__


@testset "struct" begin
    
    function test_grid_iterator()
    	g = Grid(1:3.0, 10:12.0)
    	for (val, index) in StoOpt.run(g, enumerate=true)
    		if index == (2, 3) && val == (2.0, 12.0)
    			return 1
    		end
    	end
    	return nothing
    end

    function test_noise_iterator_1d()
    	w = reshape(collect(1:6.0), 3, 2)
    	pw = ones(3, 2)*0.5
    	noise = NNoise(w, pw)
    	for (val, proba) in StoOpt.run(noise, 3)
    		if (val[1], proba) == (6.0, 0.5)
    			return 1
    		end
    	end
    	return nothing
    end

    function test_noise_iterator_2d()
    	w = reshape(collect(1:12.0), 3, 2, 2)
    	pw = ones(3, 2)*0.5
    	noise = NNoise(w, pw)
    	for (val, proba) in StoOpt.run(noise, 3)
    		if (val, proba) == ([6.0, 12.0], 0.5)
    			return 1
    		end
    	end
    	return nothing
    end

    @test test_grid_iterator() == 1
    @test test_noise_iterator_1d() == 1
    @test test_noise_iterator_2d() == 1

end

