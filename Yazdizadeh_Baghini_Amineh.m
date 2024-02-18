%% 
% *Project assignment:*

clear all
close all
addpath ('C:\Users\Amineh\Desktop\ESTIMATION\Assignment\Functions')
%% 
% Number a sample

N=10000; %set it at a suitable amount 
%% 
% project assignment

Student = 'Amineh yazdizadeh baghini';
Matriculation = '0000998863';
[Measurements] = IdentifyThis(N, Student, Matriculation);
%% *1.1 understanding model structure:*
% for identification problem and understand the model structure, we can proceed 
% with the autocorrelation analysis of the output signal.

% Collect measurements
input = Measurements.u;
output= Measurements.y;

T = 0:0.1:N-1;  % Total time
t = 1000;   % Observing time

%Plotting Input-Output Graph
figure;
subplot(211)
plot(1:t, input(1:t), 'g')
title('Input')
xlabel('Time')
ylabel('Amplitude')
subplot(212)
plot(1:t, output(1:t), 'b')
title('Output')
xlabel('Time')
ylabel('Amplitude')
%% 
% our input plot is zero for all time (\(u(t) = 0\) for all \(t\)), it means 
% that our system is not being influenced by any external input. In the context 
% of system identification and modeling,we are dealing with an AutoRegressive 
% (AR) model.
% 
% An AR model describes a time series as a linear combination of its past values. 
% The general form of an AR(p) model is given by:
% 
% 
% 
% $$y\left(t\right)=a_1 y\left(t-1\right)+a_2 y\left(t-2\right)+\ldots+a_p y\left(t-p\right)+\omega 
% \left(t\right)$$
% 
% 
% 
% where $a_1 ,a_2 ,\ldotp \ldotp \ldotp ,a_p \;$are the coefficients,$y\left(t-p\right)$represents 
% past values of the output, and $\omega \left(t\right)$ is the error term.
% 
% 
% 
% In the case of an AR model, the output at any given time $y\left(t\right)$ 
% is influenced by its own past values, and in the absence of an external input 
% $u\left(t\right)=0$, the system's behavior is determined solely by its own dynamics.
% 
% 
% 
% To confirm this, you might want to analyze the autocorrelation function of 
% the output signal. If the autocorrelation function shows a significant correlation 
% with past values, it supports the idea of an AR model.

%Autocorrelation of Output Signal

autocorr_output = autocorr(output, 10);

% Plotting autocorrelation
figure;
stem(0:10, autocorr_output , ".");
title('Autocorrelation of Output');
xlabel('Lag Time');
ylabel('NACF');
%% 
% our autocorrelation plot decays and is near zero around time lag 8, it shows 
% that there is little correlation between the current output value of the time 
% series model and its value at lag 8.
% 
% 
% 
% In the context of system identification, a rapidly decaying autocorrelation 
% function is a characteristic of a system that might be well-represented by an 
% AutoRegressive (AR) model. where the input is zero for all time, the output's 
% behavior is likely determined by its own dynamics, because the output autocorrelation 
% plot which decays quickly, it implies that the current value of the output is 
% not strongly correlated with values from the recent past, which aligns with 
% the behavior of an AR model. 
% 
% The lag at which the autocorrelation becomes near zero (in this case, lag 
% 8) can be an important parameter when determining the order of your AR model 
% if you decide to use one for modeling. it suggests that including values beyond 
% this lag may not contribute significantly to explaining the current value of 
% the time series. Therefore, a reasonable initial guess for the complexity (order) 
% of your AR model might be around 8. ofcourse it need to be determined more precisely 
% based on statistical hypothesis test or criteria, such as F-test or FPE(Final 
% Prediction Error), information criteria like AIC (Akaike Information Criterion) 
% or MDL (Minimal Discription Length), or through other model complexity selection 
% like using validation set.
% 
% 
%% *1.2 Define the required estimation algorithm:*
% 
% 
% by analysis of input, output and autocorrelation plot, we understood almost 
% likely we are dealing with an Autoregressive model of order n=8. in AR model, 
% complexity p is equal to n; our order.
% 
% $$y\left(t\right)=-a_1 y\left(t-1\right)-a_2 y\left(t-2\right)-\ldots-a_p 
% y\left(t-p\right)+\omega \left(t\right)$$
% 
% the Linear Regression form is as follow:
% 
% $$y\left(t\right)=\varphi^T \left(t\right)\theta +w\left(t\right)$$
% 
% $\varphi \left(t\right)$ is the vector of the inputs, $\theta$ is the vector 
% of unknown parameters
% 
% $$\varphi \left(t\right)={\left\lbrack -y\left(t-1\right)\;-y\left(t-2\right)\;\;\ldotp 
% \ldotp \ldotp -y\left(t-p\right)\right\rbrack }^T$$
% 
% $$\theta ={\left\lbrack a_1 \;\;\ldotp \ldotp \ldotp \;a_p \right\rbrack }^T$$
% 
% The chosen method for parameter estimation in this problem is Least Squares. 
% The objective is to estimate the parameters for a vector of coefficients with 
% dimensions . This dimensionality contributes to the complexity of the problem. 
% The formulation of Least Squares Estimation in matrix form is expressed as follows:
% 
% $$Y=H_y \left(p\right)\theta +\varepsilon$$
% 
% $H_{y\left(\right.} \left(p\right)\;$is the Hankel matrix of order p  for 
% N sample. 
% 
% $$H_{y\left(\right.} \left(p\right)=\left\lbrack \begin{array}{ccc}-y\left(0\right) 
% & \cdots  & -y\left(1-p\right)\\-y\left(1\right) & \cdots  & -y\left(2-p\right)\\\vdots  
% & \vdots  & \vdots \\-y\left(N\right) & \cdots  & -y\left(N-p\right)\end{array}\right\rbrack$$
% 
% $\theta$ is parameters vector we want to estimate and $\varepsilon$ is called 
% residual
% 
% so the diffrence between observe output and predicted one is equal to:
% 
% $$\varepsilon \left(t\right)=Y\left(t\right)-\hat{Y} \left(t\right)=H_y \left(p\right)\theta 
% +\varepsilon -H_{y\left(\right.} \left(p\right)\hat{\theta} \;$$
% 
% The LS method aims at finding the estimate of the paramters which minimizes 
% the error
% 
% by considering the loss function as follows:
% 
% $$J\left(\theta \right)=\frac{1}{N-p\;}\sum_{t=1}^N \varepsilon {\;}^2 \left(t\right)=\frac{1}{N-p}\sum_{t=1}^N 
% {\left(Y\left(t\right)-H_{y\left(\right.} \left(p\right)\hat{\theta} \right)}^2 
% \;$$
% 
% by solving optimization problem 
% 
% $$\min_{\theta \in R^P } \;\;J\left(\theta \right)$$
% 
% obtain
% 
% $$\hat{\theta} =-{\left({{\frac{1}{N-p}H}^T }_y \left(p\right)H_y \left(p\right)\right)}^{-1} 
% {{\frac{1}{N-p}H}^T }_y \left(p\right)Y$$
% 
% such that $\hat{\theta}$ is argument that minimize our lost function
% 
% $$\hat{\theta} =\arg \;\min_{\theta \in R^P } \;\;J\left(\theta \right)$$

max_order = 20;
cost_values = zeros(1, max_order);

for numparameters = 1:max_order
    % Estimate parameters using LS
    estimated_parameters = Ls_opt_solution(output, numparameters);
    
    % Compute the cost function for the current model order
    cost_values(numparameters) = CostFunction(output, estimated_parameters);
end

% Find the model order with the minimum cost
[min_cost, min_cost_order] = min(cost_values);

% Plot the cost function over the model order or number of parameters
figure;
plot(1:max_order, cost_values, 'o-', 'LineWidth', 2);
hold on;
plot(min_cost_order, min_cost, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
title('Cost Function vs. Model Order');
xlabel('Model Order');
ylabel('Cost Function (J)');
legend('Cost Function', 'Minimum Cost');
grid on;
hold off;
%% 
% in LS method on training set, as you can see by increasing complexity the 
% lost function decrease, because optimization problem are modeling our disturbance 
% or noise.
% 
% In batch method by adding new sample we need to compute hankel matrix again 
% and comput its inverse also, and do it again by every updating data, but by 
% RLS method we dont need to compute all the past data we just compute new estimation 
% ou new parameters by adding correction value to previous value of parameters 
% 
% 
% 
% to implement a recursive estimation algorithm for an AutoRegressive (AR) model 
% without using matrix inversion:
% 
% Covariance Matrix update: 
% 
% $$P\left(t\right)=\left(P\left(t-1\right)-\varphi \left(t\right)\;\varphi^T 
% \left(t\right)\right)$$
% 
% Gain vector update:
% 
% $$K\left(t\right)={P\left(t\right)}^{-1} \varphi \left(t\right)$$
% 
% Residual vector update:
% 
% $$\varepsilon \left(t\right)=y\left(t\right)-\varphi \left(t\right)\overset{\wedge 
% }{\;\theta }$$
% 
% Parameter Update:
% 
% $$\overset{\wedge }{\theta} \left(t\right)=\overset{\wedge }{\theta} \left(t-1\right)+K\left(t\right)\;\varepsilon 
% \left(t\right)$$
% 
% 

max_order = 20;
cost_RLS= zeros(1, max_order);

for numparameters = 1:max_order
    % Estimate parameters using RLS
    [~,epsilon] = RLS_opt_solution(output, numparameters);
    
    % Compute the cost function for the current model order
    sumSquared =sum(epsilon.^2);
  
    cost_RLS(numparameters)=sumSquared/N-numparameters ;
   
end

% Find the model order with the minimum cost
[min_cost_RLS, min_cost_RLS_order] = min(cost_RLS);


% Plot the cost function over the model order or number of parameters
figure;
plot(1:max_order, cost_RLS, 'o-', 'LineWidth', 2);
hold on;
plot(min_cost_RLS_order, min_cost_RLS, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
title('Cost RLS vs. Model Order');
xlabel('Model Order');
ylabel('Cost RLS');
legend('Cost RLS', 'Minimum order');
grid on;
hold off;
%% 
% in RLS method on training set, it is the case agian.by increasing complexity 
% the lost function decrease because model our noise also.
%% 
%% *1.3 Estimate the model order*
% *the question is which order is more appropriate ?*
% 
% Model complexity selection is a crucial aspect of building effective models, 
% especially in the field of system identification and statistical modeling. The 
% more important reasons for considering model complexity selection include:
%% 
% # Overfitting and Generalization:
%% 
% * Overfitting occurs when a model is too complex and captures noise or random 
% fluctuations in the training data, rather than the underlying patterns.
% * A model that is too complex may perform well on the training data but generalize 
% poorly to new, unseen data.
%% 
% # Balance between Fit and Simplicity:
%% 
% * There is a trade-off between fitting the training data well and keeping 
% the model simple.
% * More complex models can fit the training data better, but they may not generalize 
% well to new data. Simpler models are often more robust and generalize better.
%% 
% # Occam's Razor Principle:
%% 
% * Occam's Razor, a principle in philosophy and science, suggests that among 
% competing hypotheses, the one with the fewest assumptions should be selected.
% * Applied to modeling, it implies that simpler models are preferred unless 
% there is clear evidence that a more complex model is necessary.
%% 
% some of model complexity selection criteria are :
%% 
% # Validation Set
% # F_test (on training set)
% # FPE; Final Prediction Error (on training set)
% # AIC; Akaike Information Criterion (on training set)
% # MDL; Minimum Description Length (on training set)
%% 
% Final Prediction Error:
% 
% This method adds a penalty term to the cost function, higher orders J decreases. 
% its as follows:
% 
% $$\textrm{FPE}=\frac{N+p}{N-p}\;\log \;J\left(\overset{\wedge }{\theta} \right)$$

max_order=20;
FPE=zeros(1,max_order);
orders=zeros(1,4);

for numparameters = 1:max_order
    estimated_parameters = Ls_opt_solution(output, numparameters);
    FPE(numparameters)= final_predict_error(output,estimated_parameters);
end

% Find the model order with the minimum cost
[min_FPE, min_FPE_order] = min(FPE);

% Plot the cost function over the model order or number of parameters
figure;
plot(1:max_order, FPE, 'o-', 'LineWidth', 2);
hold on;
plot(min_FPE_order, min_FPE, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
title('FPE vs. Model Order');
xlabel('Model Order');
ylabel('FPE');
legend('FPE', 'Minimal order');
grid on;
hold off;
fprintf('Opt_Order through FPE: %d\n', min_FPE_order);
orders(1)=min_FPE_order;
%% 
% 
% 
% Akaike Information Criterion:
% 
% In summary, AIC is a widely used criterion for model selection that considers 
% both the goodness of fit and the complexity of the model, providing a balance 
% to avoid overfitting.
% 
% $$\textrm{AIC}=N\;\log \;J\left(\overset{\wedge }{\theta} \right)+2\;p$$

max_order=20;
AIC=zeros(1,max_order);

for numparameters = 1:max_order
    estimated_parameters = Ls_opt_solution(output, numparameters);
    AIC(numparameters)= final_predict_error(output,estimated_parameters);
end

% Find the model order with the minimum cost
[min_AIC, min_AIC_order] = min(AIC);

% Plot the cost function over the model order or number of parameters
figure;
plot(1:max_order, AIC, 'o-', 'LineWidth', 2);
hold on;
plot(min_AIC_order, min_AIC, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
title('AIC vs. Model Order');
xlabel('Model Order');
ylabel('AIC');
legend('AIC', 'Minimal order');
grid on;
hold off;
fprintf('Opt_Order through AIC: %d\n', min_AIC_order);
orders(2)=min_AIC_order;
%% 
% 
% 
% It can be proven that for $N\to \infty$, FPE and AIC are asymptotically equivalent. 
% Even if $N$ is very large, the risk of overfitting is not negligible and there 
% is the probability that AIC and FPE will return a model order $n$ which is greater 
% than the order of the real model.
% 
% Minimum Description Length :
% 
% This method consists in adding a penalty term to the logarithm of the loss 
% function such that 
% 
% $$f(N, P) = kpg(N)$$
% 
% $$lim_{N->infinity} \quad g(N)= infinity$$
% 
% $$lim_{N->infinity} \quad g(N)/N= 0$$
% 
% Namely, g(N) must go to infinity slowly with respect to N, so we can choose 
% 
% $$\textrm{MDL}=N\;\log \;J\left({\overset{\wedge }{\theta} }_N \right)+2\;p\;\log 
% \left(N\right)$$

max_order=20;
MDL=zeros(1,max_order);

for numparameters = 1:max_order
    estimated_parameters = Ls_opt_solution(output, numparameters);
    MDL(numparameters)= minimum_description_length(output,estimated_parameters);
end

% Find the model order with the minimum cost
[min_MDL, min_MDL_order] = min(MDL);

% Plot the cost function over the model order or number of parameters
figure;
plot(1:max_order, MDL, 'o-', 'LineWidth', 2);
hold on;
plot(min_MDL_order, min_MDL, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
title('MDL Criterion vs. Model Order');
xlabel('Model Order');
ylabel('MDL');
legend('MDL', 'Minimal order');
grid on;
hold off;
fprintf('Opt_Order through MDL: %d\n', min_MDL_order);
orders(3)=min_MDL_order;
%% 
% 
% 
% MDL with Validation Set:

max_order = 20;
MDL_V= zeros(1, max_order);

% Split data into training and validation sets (e.g., 80% training, 20% validation)
split_ratio = 0.8;
split_index = round(split_ratio * length(output));
training_data = output(1:split_index);
validation_data = output(split_index + 1:end);

for numparameters = 1:max_order
    estimated_parameters = Ls_opt_solution(training_data, numparameters);
    
    MDL_V(numparameters) = minimum_description_length(validation_data, estimated_parameters);
end

% Find the model order with the minimum MDL on the validation set
[min_MDL_V, min_MDL_V_order] = min(MDL_V);

% Plot the MDL function over the model order
figure;
plot(1:max_order, MDL_V, 'o-', 'LineWidth', 2);
hold on;
plot(min_MDL_V_order, min_MDL_V, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
title('MDL Criterion on Validation Set');
xlabel('Model Order');
ylabel('MDL');
legend('MDL', 'Optimal Order');
grid on;
hold off;

fprintf('Opt_Order through validation set: %d\n', min_MDL_V_order);
orders(4)=min_MDL_V_order
%% 
% choose the optimal order

optimal_order=min(orders);
fprintf('Optimal Order is: %d\n', optimal_order);
%% *I checked all LS ,RLS ,RWLS algorithem for all complexity selection criterion to estimate my prameter. RLS,RWLS leads to optimal order 2 whoes whiteness test shows we dont have white noise residual, only LS lead to order 4 whose whitness test show our residual is white noise which means our order 4 is correct order for our model.*
%% *1.4 Validate the model*
% With the given complexity we obtain a model characterized by the following 
% parameters:

estimated_parameters=Ls_opt_solution(output,optimal_order)
%% 
% Now we need to evaluate the identified model to describe the process that 
% has generated the data. 
% 
% Assuming that our data is well described by linear regression model, we can 
% make the following assumptions to the residual ε(t, θ): 
% 
% 1) ε(t, θ) is a zero mean white process
% 
% 2) ε(t, θ) is uncorrelated with the input signal u(t) 
% 
% two common tests are known as: the White Noise Test and the Cross-Correlation 
% Test. These tests help validate whether the residuals satisfy the characteristics 
% of a white noise process and are uncorrelated with the input signal.
% 
% 1) White Noise Test:
% 
% Wihteness test is a binary statistical test in which the residual of the model 
% is going to be observed. If $\hat r_\epsilon$ (i.e.variance of the samples residual) 
% is a white process, then the first element of the variance vector must be equal 
% to the variance of the white process (i.e. $\hat r_\epsilon(0)$ = $\sigma^2_w$ 
% ) and all other elements equal to zero, based on the theory.
% 
% $\hat r_\epsilon(0)$$\to$$\sigma^2_w$ ,$\;N\to \infty$
% 
% $\hat r_\epsilon(\tau)$$\to$$0$ ,$\forall \tau \not= 0$, $N\to \infty$
% 
% In practice, the elements rather than the first one must converge to zero. 
% Therefore, if the estimation were being performed well, It would be anticipated 
% that for any non-zero value of the the time lags, the autocorrelation function 
% converge to zero.
% 
% 

numsamples=length(output);
H=-Hankel(output,optimal_order);
residuals=(output(optimal_order+1:end)-H*estimated_parameters);
%% 
% Now let's plot the autocorrelation

%Autocorrelation of residuals

autocorr_residuals = autocorr(residuals, 10);

% Plotting autocorrelation
figure;
stem(0:10, autocorr_residuals , ".");
title('Autocorrelation of residuals');
xlabel('Lag Time');
ylabel('NACF');
%% 
% 
% 
% do hypothesis test

% Assuming residuals are stored in the variable 'residuals'
[h, p, ~] = lbqtest(residuals, 'lags', [1, 2, 3, 4,5,6,7,8,9,10]);
disp(['White Noise Test p-values: ', num2str(p)]);
%% 
% The interpretation of these p-values is crucial for determining whether the 
% residuals exhibit white noise properties. If the p-values are greater than a 
% chosen significance level (e.g., 0.05), it suggests that there is no evidence 
% to reject the null hypothesis, indicating that the residuals are consistent 
% with white noise.
% 
% 
% 
% |so as you can see our residual of our prediction is white noise. because 
% p_value are grater than 0.05|
%% *I checked all LS ,RLS ,RWLS algorithem for all complexity selection criterion to estimate my prameter. RLS,RWLS leads to optimal order 2 whoes whiteness test shows we dont have white noise residual, only LS lead to order 4 whose whitness test show our residual is white noise which means our order 4 is correct order for our model.*
% 
% 
% 2) Cross Correlation test: 
% 
% we need to compute the cross correlation between ε(t, θ) and u(t) and see 
% if they are completely uncorrelated:

%crosscorrelation of residuals with input signal

crosscor_r_u = crosscorr(residuals, input);

% Plotting crosscorrelation
figure;
stem( crosscor_r_u, ".");
title('crosscorrelation of residuals with input signal');
xlabel('Lag Time');
ylabel('cross correlation');
%% 
% ε(t, θ) and u(t) are indeed completely uncorrelated (actually we already knew 
% this because we are working with a system with no input). 
% 
% our plot shows that our residuals and our input signal is completely uncorrelated
