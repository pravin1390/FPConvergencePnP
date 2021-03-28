function u = prox_despeckling(lambda,D,M,y,z)
% Proximal operator of the despeckling data fidelity function
% lambda = Penalty parameter
% D = Normalizing coefficients
% M = No. of looks
% y = Observed log-intensity image
% x = Input argument (image)
u = y;              % Initial value = Observed log-intensity image
for k = 1:10        % No. of Newton's method iterations
    num = D.*(u-z) + (M/lambda)*(1 - exp(y-u)); % Derivative
    den = D + (M/lambda)*exp(y-u);              % Second derivative
    u = u - num./den;
end
end
