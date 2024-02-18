function [rec_esti_parameters,epsilon] = RLS_opt_solution(output, numparameters)

    numunbias = length(output) - numparameters;

    observedoutput = output(numparameters + 1:end);

    % Initialize RLS algorithm variables
    P = eye(numparameters);
    parameter= zeros(numparameters, 1);

    % RLS algorithm without forgetting factor
    for t = numparameters+1:numunbias
        phi_t = -output(t-1:-1:t-numparameters);  % Past values
        K_t = P * phi_t / (1 + phi_t' * P * phi_t);
        epsilon_t = observedoutput(t) - phi_t' * parameter;
        
        % Update parameter estimate
        parameter = parameter + K_t * epsilon_t;
        
        % Update covariance matrix
        P = (P - K_t * phi_t' * P) / (1 + phi_t' * P * phi_t);
       
    end

    rec_esti_parameters = parameter;
    epsilon = output - phi_t' * rec_esti_parameters;
    
    
    
end

