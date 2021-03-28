function f = eval_fidelity(x,M,y)
%EVAL_FIDELITY Data fidelity term for SAR despeckling
% x = Input argument
% M = No. of looks
% y = Observed log-intensity image

f = x + exp(y-x);
f = sum(f(:));
f = M*f;

end

