function AIC = akaike_info_cri(output,rec_esti_parameters)
 
    numsamples = length(output);
    numparameters = length(rec_esti_parameters);
  
    costj= CostFunction(output, rec_esti_parameters);
    AIC = numsamples*log(costj) + 2*numparameters;
end
