% Code for image despeckling using PnP-ADMM w.r.t. a non-standard inner
% product, as described in the following paper:
%
% P. Nair, R. G. Gavaskar and K. N. Chaudhury, "Fixed-Point and Objective 
% Convergence of Plug-and-Play Algorithms", IEEE Transactions on 
% Computational Imaging, 2021.
% Link: https://doi.org/10.1109/TCI.2021.3066053
%

clearvars; close all; clc;

res = [256,256];    % Resize image to this size (if not already so)
ground = double(imresize(imread('./peppers.png'),res));
ground(ground==0) = 1;  % To avoid taking log of 0, all values must be nonzero

M = 5;  % No. of looks

% Forward model
[rr,cc] = size(ground);
r_orig = ground.^2;             % Intensity image
noise = gamrnd(M,1/M,[rr,cc]);  % Gamma noise
s = r_orig.*noise;              % Observed intensity image
y = log(s);                     % Observed log-intensity image

figure; imshow(uint8(sqrt(s))); title('Observed (speckled) image');
drawnow;

%% Warm start
% For the first few iterations we allow the denoiser weights to be computed
% from the current image, instead of fixing the weights. This makes the
% denoiser non-linear but allows us to obtain a good guide image which can
% be used to fix the weights later. We use a different set of parameters
% for these first few iterations, in order to get better results.
srad_warm = 2;          % Search window radius for warm start
prad_warm = 2;          % Patch radius for warm start
h_warm = (50/255)*log(255^2);% Standard deviation of Gaussian for warm start
warm_iters = 5;         % No. of iterations for warm start
rho_warm = 5;           % Penalty parameter for warm start

W_warm = @(x) JNLM(x,x,prad_warm,srad_warm,h_warm);     % Nonlinear NLM denoiser
prox_op = @(z,r) prox_despeckling(r,ones(rr,cc),M,y,z); % Proximal operator w.r.t. standard inner product
fprintf('Warm start:\n');
y_init = y * M / exp(psi(M));   % Initialization; helpful for improving results
x0 = pnpADMM(y_init,0,rho_warm,prox_op,W_warm,[],-1,warm_iters,[]); % Run warm-start iterations
fprintf('\n');

%% Main iterations
% We now compute the denoiser weights from the image obtained from the warm
% start. These weights are kept fixed in the rest of the iterations, so the
% denoiser is linear.
srad = 5;                   % Seach window radius
prad = 1;                   % Patch radius
h = (55/255)*log(255^2);    % Standard deviation of Gaussian
maxiters = 100;             % Max. no. of iterations
rho = 0.2;                  % Penalty parameter

W_lin = @(x) JNLM(x,x0,prad,srad,h);    % Linear NLM denoiser
[~,~,~,~,D] = JNLM(x0,x0,prad,srad,h);  % Normalizing coefficients (D, see eq. (12) in the paper)
prox_scaled = @(z,r) prox_despeckling(r,D,M,y,z); % Proximal operator w.r.t. inner product defined by D
objfun = @(xk,yk,vk) eval_fidelity(yk,M,y) + ...
         rho*eval_regularizer(yk,vk,D); % Function to calculate objective values

fprintf('Main iterations:\n');
[x_hat_scaled,~,~,~,~,y_err,objval] = ...
    pnpADMM(x0,0,rho,prox_scaled,W_lin,[],-1,maxiters,objfun); % Run PnP-ADMM

r_hat_scaled = sqrt(exp(x_hat_scaled));   % Recovered amplitude image

% Print metrics
fprintf('PSNR (linear) = %f\n',psnr(uint8(r_hat_scaled),uint8(ground)));
fprintf('SSIM (linear) = %f\n',ssim(uint8(r_hat_scaled),uint8(ground)));

% Display recovered image
figure; imshow(uint8(r_hat_scaled));
title('Recovered image');

% Plot iteration-wise residues and objective values
figure('Units','Normalized','Position',[0.2,0.2,0.6,0.6]);
subplot(1,2,1);
plot(y_err,'Linewidth',2.5); grid on; axis tight;
xlabel('Iterations, $k$','Interpreter','latex');
title('Errors, $\| y_{k+1}-y_k \|_2$','Interpreter','latex');
subplot(1,2,2);
plot(objval,'Linewidth',2.5); grid on; axis tight;
xlabel('Iterations, $k$','Interpreter','latex');
title('Objective values, $f(x_k) + \rho h_D(y_k)$','Interpreter','latex');
drawnow;

