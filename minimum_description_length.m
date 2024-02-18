function MDL=minimum_description_length(output,rec_esti_parameters)

    numparameters = length(rec_esti_parameters);
    numsamples= length(output);
    costj=CostFunction(output,rec_esti_parameters);

    %penalty term is given by 2nlog(N)
    MDL= numsamples*log(costj) + 2*numparameters*log(numsamples);
end