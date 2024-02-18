function Hankel = Hankel(output,numparameters)



    numsamples= length(output);
    Hankel= zeros(numsamples-numparameters,numparameters);
    
    % Computing the Hankel Matrix Row-by-Row
    for k=1:numsamples-numparameters
        Hankel(k,:) = output(k+numparameters-1:-1:k);
    end

    
end