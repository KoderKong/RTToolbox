%% Inner product, D1 scalar
rng default
sparsor = @(arg) tensor(sparse1(arg));
[a,i] = sparsor(rand(1,1,9));
b = sparsor(rand(1,1,9));
c = sparsor(rand(1,1,9));
x = a(i)*b(i)*c(~i);
assert(isequal(degree(x),0))
assert(isequal(size(x),[1 1]))
ey = entry(a).*entry(b).*entry(c);
assert(isequal(entry(x),sum(ey)))
%% Entrywise product, D1 scalar
rng default
sparsor = @(arg) tensor(sparse1(arg));
[a,i] = sparsor(rand(1,1,9));
b = sparsor(rand(1,1,9));
c = sparsor(rand(1,1,9));
y = a(i)*b(i)*c(i);
assert(isequal(index(y),i))
assert(isequal(size(y),[1 1 9]))
ey = entry(a).*entry(b).*entry(c);
assert(isequal(entry(y),ey))
%% Outer product, D1 scalar
rng default
sparsor = @(arg) tensor(sparse1(arg));
[a,i] = sparsor(rand(1,1,9));
[b,j] = sparsor(rand(1,1,8));
[c,k] = sparsor(rand(1,1,7));
z = a(i)*b(j)*c(k);
assert(isequal(index(z),[i j k]))
assert(isequal(size(z),[1 1 9 8 7]))
eb = shiftdim(entry(b),-1);
ec = shiftdim(entry(c),-2);
ez = entry(a).*eb.*ec;
assert(isequal(entry(z),ez))
%% Entrywise relation, D2 scalar
rng default
sparsor = @(arg) tensor(sparse1(arg));
[y,i,j] = sparsor(rand(1,1,9,9));
assert(~isequal(y(i,~j),y(~j,i)))
assert(isequal(entry(y(i,~j)),entry(y)))
assert(isequal(entry(y(~j,i)),entry(y)))
%% Permute and copy, D2 scalar
rng default
sparsor = @(arg) tensor(sparse1(arg));
[y,i,j] = sparsor(rand(1,1,9,8));
z(i,~j) = y(~j,i);
assert(isequal(z(i,~j),y(~j,i)))
ez = permute(entry(y),[1 2 4 3]);
assert(isequal(entry(z),ez))
%% Left division, D2 scalar
rng default
sparsor = @(arg) tensor(sparse1(arg));
[A,l,lp] = sparsor(rand(1,1,9,9));
[b,i,~] = sparsor(rand(1,1,8,9));
u(i,~l) = A(l,lp)\b(i,lp);
assert(isequal(size(u),[1 1 8 9]))
eA = squeeze(entry(A)).';
eb = squeeze(entry(b)).';
eu = squeeze(entry(u)).';
assert(isequal(eu,eA\eb))
%% Mixed product, D2 scalar
rng default
sparsor = @(arg) tensor(sparse1(arg));
[A,l,lp] = sparsor(rand(1,1,9,9));
[u,i,~] = sparsor(rand(1,1,8,9));
b(i,lp) = A(l,lp)*u(i,~l);
assert(isequal(size(b),[1 1 8 9]))
eA = squeeze(entry(A)).';
eu = squeeze(entry(u)).';
eb = squeeze(entry(b)).';
assert(isequal(eb,eA*eu))
%% Outer addition, D1 scalar
rng default
sparsor = @(arg) tensor(sparse1(arg));
[c,j] = sparsor(rand(1,1,9));
[x,i] = sparsor(rand(1,1,8));
l(i,j) = log(c(j)+x(i));
ec = shiftdim(entry(c),-1);
el = log(ec+entry(x));
assert(isequal(entry(l),el))
%% Mixed product, D2 matrix
rng default
sparsor = @(arg) tensor(sparse1(arg));
[A,i,~] = sparsor(rand(9,5,7,6));
[B,~,j] = sparsor(rand(5,8,7,6));
C = A(i,~j)*B(i,~j);
assert(isequal(index(C),[i ~j]))
assert(isequal(size(C),[9 8 7 6]))
eC = pagemtimes(entry(A),entry(B));
assert(isequal(entry(C),eC))
%% Trace contraction, D2 matrix
rng default
sparsor = @(arg) tensor(sparse1(arg));
[C,i,~] = sparsor(rand(9,9,8,8));
t = trace(C(i,~i));
assert(isequal(degree(t),0))
eC = entry(C);
et = 0;
for k = 1:8
    et = et+trace(eC(:,:,k,k));
end
assert(isequal(entry(t),et))
%% Diagonal attraction, D2 matrix
rng default
sparsor = @(arg) tensor(sparse1(arg));
[C,i,~] = sparsor(rand(9,9,8,8));
d = diag(C(i,i));
assert(isequal(index(d),i))
eC = entry(C);
ed = zeros(9,1,8);
for k = 1:8
    ed(:,:,k) = sparse(diag(eC(:,:,k,k)));
end
assert(isequal(entry(d),ed))
%% Inner product, DN vector
rng default
sparsor = @(arg) tensor(sparse1(arg));
dims = [9 1 8:-1:4];
[x,k] = sparsor(randn(dims,'like',1j));
y = x(~k)'*x(~k);
assert(isequal(degree(y),0))
ex = entry(x);
ey = ex(:)'*ex(:);
assert(isequal(entry(y),ey))
assert(isreal(entry(y)))
%% Column concatenation, D1 vector
rng default
sparsor = @(arg) tensor(sparse1(arg));
M = 9;
N = 8;
[c,i] = sparsor(randn(M,1,M-2));
[~,~] = sparsor(randn(N,1,N-2));
C = [ones(M,1) abs(c(i))];
assert(isequal(index(C),i))
eC = ones(M,2,M-2);
eC(:,2,:) = full(abs(entry(c)));
assert(isequal(entry(C),eC))
%% Row concatenation, D1 vector
rng default
sparsor = @(arg) tensor(sparse1(arg));
M = 9;
N = 8;
[~,~] = sparsor(randn(M,1,M-2));
[b,j] = sparsor(randn(N,1,N-2));
B = [b(j).'; ones(1,N)];
assert(isequal(index(B),~j))
eB = ones(2,N,N-2);
eB(1,:,:) = full(permute(entry(b),[2 1 3]));
assert(isequal(entry(B),eB))
%% Outer relation, D1 vector
rng default
sparsor = @(arg) tensor(sparse1(arg));
M = 9;
N = 8;
[c,i] = sparsor(randn(M,1,M-2));
[b,j] = sparsor(randn(N,1,N-2));
A(i,~j) = b(j).'+abs(c(i));
ToF = A(i,~j) >= b(j).';
assert(isequal(index(ToF),[i ~j]))
assert(isequal(size(ToF),[M N M-2 N-2]))
assert(isa(entry(ToF),'logical'))
assert(all(entry(ToF),'all'))
%% Index concatenation, D2 matrix
rng default
sparsor = @(arg) tensor(sparse1(arg));
[A,i,j] = sparsor(rand(9,8,7,6));
[B,~,k] = sparsor(rand(9,8,6,4));
C = cat(k,A(i,j),B(j,k));
assert(isequal(index(C),[i j k]))
assert(isequal(size(C),[9 8 7 6 5]))
eB = reshape(entry(B),[9 8 1 6 4]);
eC = cat(5,entry(A),repmat(eB,[1 1 7]));
assert(isequal(entry(C),eC))
%% Example 1, doc rref
A = magic(3);
sA = sparse1(A);
[sRA,~,q] = rref(sA);
RA = rref(A(:,q));
assert(isequal(RA,sRA))
%% Example 2, doc rref
B = magic(4);
sB = sparse1(B);
[sRB,sp,q] = rref(sB);
[RB,p] = rref(B(:,q));
assert(norm(RB-sRB,'fro') < 1e-15)
assert(isequal(p,sp))
%% Example 3, doc rref
A = magic(3);
A(:,4) = [1; 1; 1];
sA = sparse1(A);
[sR,~,q] = rref(sA);
R = rref(A(:,q));
assert(norm(R-sR,'fro') < 1e-16)
%% Example 4, doc rref
A = [magic(3) eye(3)];
sA = sparse1(A);
[sR,~,q] = rref(sA);
R = rref(A(:,q));
assert(norm(R-sR,'fro') < 1e-16)
%% Example 5, doc rref
A = [1  1  5;
    2  1  8;
    1  2  7;
    -1  1 -1];
b = [6 8 10 2]';
M = [A b];
sM = sparse1(M);
[sR,~,q] = rref(sM);
R = rref(M(:,q));
assert(norm(R-sR,'fro') < 1e-15)
%% Example 1, doc null
A = ones(3);
sA = sparse1(A);
[sx2,q] = null(sA);
x2 = null(A(:,q),'rational');
assert(isequal(x2,sx2))
%% Example 3, doc null
A = [1 8 15 67; 7 14 16 3];
sA = sparse1(A);
[sN,q] = null(sA);
N = null(A(:,q),'rational');
assert(norm(N-sN,'fro') < 1e-5)
assert(norm(A(:,q)*N,'fro') > 0)
assert(norm(A(:,q)*sN,'fro') == 0)
