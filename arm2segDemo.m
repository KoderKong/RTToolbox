%%
% mex -R2018 -Drrlu spmex1.c
% mex -R2018 -Dbsx spmex1.c
mex -R2018 spmex1.c
format compact
%%
clear
a = arm2seg(1/3,4/3);
[Qv,Qu,r,Id] = qr(a,2);
assert(nnz(r) >= 174)
%%
I = speye(size(Qu,2));
assert(isequal(Id(:,:,end),I))
assert(isequal(nnz(Id),nnz(I)))
iseq = @(a,b,n) isequal( ...
    round(a,n),round(b,n));
Qu0 = Qu(:,:,end);
assert(iseq(Qu0'*Qu0,I,3))
[vs,~,k] = vsigma(Qv,Qu');
assert(iseq(Qv*Qu'*vs,Id(k),2))
%%
% a = arm2seg(1/3,4/3);
% r = qr(a,2); % 400 iterations
r(abs(r) < 1e-15) = 0;
assert(isequal(nnz(r),118))
x1Qr = roots(r(end,1,index(r)));
x1GB = roots([68/9 -68/27 -287/81]);
assert(iseq(x1Qr,x1GB,15))
%%
clear
[a,Qu,Qv,Id] = arm2seg(1/3,4/3);
[Qu,Qv,R] = qq(Qu,Qv); % Transform
r = Qu'*a*vsigma(Qu',a);
r(abs(r) < 1e-16) = 0;
assert(isequal(nnz(r),18))
%%
I = speye(size(Qu,2));
assert(isequal(Id(:,:,end),I))
assert(isequal(nnz(Id),nnz(I)))
iseq = @(a,b,n) isequal( ...
    round(a,n),round(b,n));
Qu0 = Qu(:,:,end);
assert(iseq(Qu0'*Qu0,I,15))
[vs,~,k] = vsigma(Qv,Qu');
assert(iseq(Qv*Qu'*vs,Id(k),15))
