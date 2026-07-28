%%
% mex -R2018a -Drrlu spmex1.c
% mex -R2018a -Dbsx spmex1.c
mex -R2018a spmex1.c
%%
clear
format compact
a = arm2seg(1/3,4/3);
[Qv,Qu,r,Id] = qr(a,2);
% [Qv,Qu,r,Id] = qr(a,2,'MaxIterations',Inf);
%%
I = speye(size(Qu,2));
assert(isequal(Id(:,:,end),I))
assert(isequal(nnz(Id),nnz(I)))
Qu0 = Qu(:,:,end);
[vs,~,k] = vsigma(Qv,Qu');
iseq = @(a,b,n) isequal( ...
    round(a,n),round(b,n));
assert(iseq(Qu0'*Qu0,I,3))
assert(iseq(Qv*Qu'*vs,Id(k),2))
% assert(iseq(Qu0'*Qu0,I,6)) % 1226 iterations
% assert(iseq(Qv*Qu'*vs,Id(k),4)) % 1226 iterations
%%
disp(nnz(r))
r(abs(r) < 1e-15) = 0;
assert(nnz(r) == 118)
x1Qr = roots(r(end,1,index(r)));
x1GB = roots([68/9 -68/27 -287/81]);
assert(iseq(x1Qr,x1GB,15))
%%
r = qr(a,2,'MaxIterations',0);
disp(nnz(r))
r(abs(r) < 1e-15) = 0;
assert(nnz(r) == 19)
x1Qr = roots(r(end,1,index(r)));
assert(iseq(x1Qr,x1GB,15))
%%
[Qv,Qu,~,Id] = qr(a,2,'MaxIterations',0);
Qu0 = Qu(:,:,end);
[vs,~,k] = vsigma(Qv,Qu');
assert(iseq(Qu0'*Qu0,I,15))
assert(~iseq(Qv*Qu'*vs,Id(k),0))
%%
[~,Qu,Qv,Id] = arm2seg(1/3,4/3);
Qu0 = Qu(:,:,end);
[vs,~,k] = vsigma(Qv,Qu');
assert(~iseq(Qu0'*Qu0,I,0))
assert(iseq(Qv*Qu'*vs,Id(k),15))
%%
[a,Qu,Qv,Id] = arm2seg(1/3,4/3);
[Qu,Qv,R] = qq(Qu,Qv);
r = Qu'*a*vsigma(Qu',a);
disp(nnz(r))
r(abs(r) < 1e-15) = 0;
assert(nnz(r) == 18)
%%
Qu0 = Qu(:,:,end);
[vs,~,k] = vsigma(Qv,Qu');
assert(iseq(Qu0'*Qu0,I,15))
assert(iseq(Qv*Qu'*vs,Id(k),15))
%%
tic
r = qr(a,2,@(A) null(A,'rational'));
toc
