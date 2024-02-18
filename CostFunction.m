function costj = CostFunction(Output, estimated_parameters)


    numsamples = length(Output);                            % consider N samples of output y
    numparameters = length(estimated_parameters);           % Number of parameters in the model

    % Create the Hankel Matrix of the Observed Output with N (number of
    % samples) rows and n(number of parameters)
    hankelMatrix = -Hankel(Output, numparameters);

    % Extract the relevant portion of the observed output from the (n+1)-th element until the end
    observedOutput = Output(numparameters + 1:end);

    % Compute the Residual
    residual = observedOutput - hankelMatrix * estimated_parameters;

    % Evaluate the Least Squares Cost Function
    costj = 1 / (numsamples - numparameters) * (residual' * residual);

end
