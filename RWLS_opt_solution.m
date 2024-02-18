function [estimated_parameters, epsilon] = RWLS_opt_solution(output, numparameters)
    
    lambda=0.5;
    numunbias = length(output) - numparameters;
    observedoutput = output(numparameters + 1:end);

    % Initialize RWLS algorithm variables
    P = eye(numparameters);
    parameter = zeros(numparameters, 1);

    % RWLS algorithm with forgetting factor
    for t = numparameters+1:numunbias
        phi_t = -output(t-1:-1:t-numparameters);  % Past values
        K_t = P * phi_t / (lambda + phi_t' * P * phi_t);
        epsilon_t = observedoutput(t) - phi_t' * parameter;
        
        % Update parameter estimate
        parameter = parameter + K_t * epsilon_t;
        
        % Update covariance matrix
        P = (1/lambda) * (P - K_t * phi_t' * P);
    end

    estimated_parameters = parameter;
    epsilon = observedoutput - phi_t' * estimated_parameters;
end
