function estimated_parameters = Ls_opt_solution(output, numparameters)
 
    numunbias= length(output) - numparameters;

    observedoutput= output(numparameters+1:end);

    hankelMatrix = -Hankel(output, numparameters);

    estimated_parameters = pinv(hankelMatrix'*hankelMatrix/numunbias)*hankelMatrix'*observedoutput/numunbias;

end