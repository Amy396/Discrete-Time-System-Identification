function FPE = final_predict_error(output, rec_esti_parameters)
     
    numsamples=length(output);
    numparameters=length(rec_esti_parameters);

    costj=CostFunction(output,rec_esti_parameters);
    FPE=((numsamples+numparameters)/(numsamples-numparameters))*costj;
end
