function u = GSSP(w,lambda,k)
% function. Let Omega denote the simplex: 
% Omega = {x in R^n: x_i >= 0 and sum x_i = lambda } and let Sigma_k denote
% the set of all k-sparse vectors. This functions projects w on to Omega
% intersect Sigma_k. Based on Kyrillidis et al 'Sparse Projections on to 
% the Simplex' 2013, JMLR.
% INPUT: vector x and (scalar) parameter lambda and sparsity k
% OUTPUT: w in simplex.
 
function v2 = Plambda(v1,sigma)
    %fprintf('v1_0 = %2.6f v1_1 = %2.6f \n',v1(1),v1(2));
    [vtemp,inds] = sort(v1,'descend');
    %fprintf('vtemp = %2.6f vtemp = %2.6f \n',vtemp(1),vtemp(2));
    rho = 0;
    StopCond = 1;
    j = 1;
    while StopCond && j <= length(vtemp)
        tau = (sum(vtemp(1:j)) - sigma)/j;
        if vtemp(j) > tau
            rho = rho +1;
            j = j +1;
        else
            StopCond = 0;
        end
    end
    tau = (sum(vtemp(1:rho)) - sigma)/rho;
    %fprintf('tau = %2.6f\n', tau);
    vtemp2 = bsxfun(@max,vtemp - tau,0);
    %fprintf('vtemp2 = %2.6f vtemp2 = %2.6f \n',vtemp2(1),vtemp2(2));
    v2 = zeros(size(v1));
    v2(inds) = vtemp2;
    %fprintf('v2 = %2.6f v2 = %2.6f \n',v2(1),v2(2));

end
%fprintf('x0 = %2.6f x1 = %2.6f \n',w(1),w(2));
N = length(w);
[~,indices] = sort(w,'descend');
S = indices(1:k);
u = zeros(N,1);
%fprintf('w_s_0 = %2.6f w_s_1 = %2.6f \n',ws(1),ws(2));
utemp = Plambda(w(S),lambda);
%fprintf('u0 = %2.6f u1 = %2.6f \n',utemp(1),utemp(2));
u(S) = utemp;
%fprintf('u0 = %2.6f u1 = %2.6f \n',u(1),u(2));

end


