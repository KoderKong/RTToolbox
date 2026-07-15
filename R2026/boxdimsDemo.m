%%
% mex -R2018a -Drrlu spmex1.c
% mex -R2018a -Dbsx spmex1.c
mex -R2018a spmex1.c
%%
clear
format compact
a = boxdims(1,0.5,0.01);
disp(a)
%%
r = qr(a,2,'Display','off');
assert(degree(r) == degree(a)+2)
disp(r)
%%
[Qu,r] = qr(a,2,'Display','iter');
iseq = @(a,b,n) isequal( ...
    round(a,n),round(b,n));
[vs,~,k] = vsigma(Qu',a);
assert(iseq(r(k),Qu'*a*vs,15))
%%
[Qv,Qu,r] = qr(a,2,'Display','off');
Qr = simplify(Qv*r); % 'mindeg'
assert(isequal(a,Qr(index(a))))
[vs,~,k] = vsigma(Qu',a);
assert(isequal(r(k),Qu'*a*vs))
%%
r = simplify(r);
disp(r)
assert(degree(r) == degree(a))
%%
k = index(r);
x3 = roots(r(end,1,k));
x = sort(x3,'descend') %#ok<NOPTS>
polyval(a,x,1)
%%
symx = sym('x',[3 1]);
symr = polyval(r,symx,1);
syma = polyval(a,symx,1);
simplify(symr.'*diag([1 -2 100])- ...
    gbasis(syma.','MonomialOrder','lex'))
%%
U = polyval(Qu',symx,1); % symU
V = polyval(Qv,symx,1); % symV
I = eye(3,3,'sym'); % symI
assert(isequal(simplify(V*U),I))
%%
tic
r = qr(a,2,@(A) null(A,'rational'));
toc
