function [x_curr,converged,iters,P,primal_residue,dual_residue,obj,z_curr] = ...
                    pnpADMM(z0,u0,rho,prox_f,W,I,tol,maxiters,objfun)
%PNPADMM: Plug-and-play ADMM
% NOTE: The notational convention for the ADMM variables in this code is
% slightly different from the paper. Please refer to the below table.
%   Notation in paper (ref. Sec. II-C)   ---->   Notation in this code
%   ==================================           =====================
%         x_k (primal variable 1)                       x_curr
%         y_k (primal variable 2)                       z_curr
%         z_k (dual variable)                           u_curr
% 
% Input arguments:
% z0 = Initial point (primal variable 2)
% u0 = Initial point (dual variable)
% rho = Penalty parameter
% prox_f = Handle to function computing the inversion step
%        (proximal map of f/rho w.r.t. the appropriate scaling matrix)
% W = Handle to denoiser
% I = Ground-truth image to compare PSNR (optional), can be empty ([])
% tol = Tolerance (optional)
% maxiters = Max. no. of iterations (optional)
%
% Output arguments:
% x_curr = Output (value of 1st primal variable)
% converged = Flag, True if algorithm converged
% iters = No. of iterations completed
% P = Iteration-wise PSNRs
% primal_residue = Iteration-wise values of norm(x_k - z_k)
% dual_residue = Iteration-wise values of norm(z_{k+1} - z_k)
% obj = Iteration-wise objective function values
% z_curr = Value of 2nd primal variable after completion
%

if(~exist('I','var') || isempty(I))
    calcPSNR = false;
else
    calcPSNR = true;
end
if(~exist('tol','var') || isempty(tol))
    tol = 1/255;
end
if(~exist('maxiters','var') || isempty(maxiters))
    maxiters = 30;
end
if(~exist('objfun','var') || isempty(objfun))
    calcObj = false;
    obj = [];
else
    calcObj = true;
end

iters = 1;
z_curr = z0;
u_curr = u0;
dual_residue = nan(1,maxiters);
primal_residue = nan(1,maxiters);
primal_residue(1) = nan;
if(calcPSNR)
    P = nan(1,maxiters);
    P(1) = psnr(z0,I,1);
end
if(calcObj)
    obj = nan(1,maxiters);
end
while(true)
    % Main algorithm
    x_next = prox_f(z_curr - u_curr,rho);
    v_next = x_next + u_curr;
    z_next = W(v_next);
    u_next = u_curr + x_next - z_next;
    
    % Calculate PSNR, residues, objective values etc.
    if(calcObj)
        obj(iters) = objfun(x_next,z_next,v_next);  % Calculate objective value
    end
    if(calcPSNR)
        P(iters+1) = psnr(z_next,I,1);              % Calculate PSNR
        fprintf('Iteration = %d,\tPSNR = %f\n',iters,P(iters+1));
    else
        fprintf('Iteration = %d\n',iters);
    end
    err = euclNorm(z_next - z_curr);
    dual_residue(iters) = err;
    primal_residue(iters+1) = euclNorm(x_next - z_next);
    if(err < tol)
        converged = true;
        break;
    end
    if(iters==maxiters)
        if(err < tol)
            converged = true;
        else
            converged = false;
        end
        break;
    end
    
    % Transition to next iteration
    iters = iters+1;
    x_curr = x_next;
    z_curr = z_next;
    u_curr = u_next;
end

if(calcPSNR)
    P(iters+1:end) = [];
else
    P = [];
end

end


function n = euclNorm(y)
% Euclidean norm of vector or matrix

n = sqrt(sum(y(:).*y(:)));

end
