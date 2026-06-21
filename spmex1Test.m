% These unit tests derive from MATLAB's lu
% and bsxfun documentation (see examples).
%% Example 3, doc lu
rng default
A = round(1000*sprand(bucky));
[L,U,p,q] = spmex1('rrlu',A);
[M,N] = size(A);
assert(isequal(size(L),[M M]))
assert(isequal(size(U),[M N]))
assert(isequal(tril(L),L))
assert(isequal(triu(U),U))
assert(isequal(diag(L),ones(M,1)))
assert(isequal(sort(p),1:M))
assert(isequal(sort(q),1:N))
[L_,U_,p_,q_] = lu(A,'vector');
ref = norm(A(p_,q_)-L_*U_,'fro');
assert(norm(A(p,q)-L*U,'fro') < ref)
%% Example 1, doc bsxfun
A = sparse([1 2 10; 3 4 20; 9 6 15]);
meanA = mean(A);
C = spmex1('bsx','minus',A,meanA);
C_ = bsxfun(@minus,A,meanA);
assert(isequal(C,C_))
stdA = std(A);
D = spmex1('bsx','rdivide',C,stdA);
D_ = bsxfun(@rdivide,C_,stdA);
assert(isequal(D,D_))
