%%
% mex -R2018 -Drrlu spmex1.c
% mex -R2018 -Dbsx spmex1.c
mex -R2018 spmex1.c
format compact
%%
clear
a = boxdims(1,0.5,0.01);
disp(a)
%%
[Qu,r] = qr(a,2,'Display','iter');
iseq = @(a,b,n) isequal( ...
    round(a,n),round(b,n));
[vs,~,k] = vsigma(Qu',a);
assert(iseq(r(k),Qu'*a*vs,15))
%%
k = index(r);
x3 = roots(r(end,1,k));
x = sort(x3,'descend');
polyval(a,x,1)
%%
clear
a = boxdims(1,0.5,0.01);
tic
r = qr(a,2,@(A) null(A,'rational'));
toc
