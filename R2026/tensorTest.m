%% Inner product, D1 scalar
rng default
[a,i] = tensor(rand(1,1,9));
b = tensor(rand(1,1,9));
c = tensor(rand(1,1,9));
x = a(i)*b(i)*c(~i);
assert(isequal(degree(x),0))
assert(isequal(size(x),[1 1]))
ey = entry(a).*entry(b).*entry(c);
assert(isequal(entry(x),sum(ey)))
%% Entrywise product, D1 scalar
rng default
[a,i] = tensor(rand(1,1,9));
b = tensor(rand(1,1,9));
c = tensor(rand(1,1,9));
y = a(i)*b(i)*c(i);
assert(isequal(index(y),i))
assert(isequal(size(y),[1 1 9]))
ey = entry(a).*entry(b).*entry(c);
assert(isequal(entry(y),ey))
%% Outer product, D1 scalar
rng default
[a,i] = tensor(rand(1,1,9));
[b,j] = tensor(rand(1,1,8));
[c,k] = tensor(rand(1,1,7));
z = a(i)*b(j)*c(k);
assert(isequal(index(z),[i j k]))
assert(isequal(size(z),[1 1 9 8 7]))
eb = shiftdim(entry(b),-1);
ec = shiftdim(entry(c),-2);
ez = entry(a).*eb.*ec;
assert(isequal(entry(z),ez))
%% Entrywise relation, D2 scalar
rng default
[y,i,j] = tensor(rand(1,1,9,9));
assert(~isequal(y(i,~j),y(~j,i)))
assert(isequal(entry(y(i,~j)),entry(y)))
assert(isequal(entry(y(~j,i)),entry(y)))
%% Permute and copy, D2 scalar
rng default
[y,i,j] = tensor(rand(1,1,9,8));
z(i,~j) = y(~j,i);
assert(isequal(z(i,~j),y(~j,i)))
ez = permute(entry(y),[1 2 4 3]);
assert(isequal(entry(z),ez))
%% Left division, D2 scalar
rng default
[A,l,lp] = tensor(rand(1,1,9,9));
[b,i,~] = tensor(rand(1,1,8,9));
u(i,~l) = A(l,lp)\b(i,lp);
assert(isequal(size(u),[1 1 8 9]))
eA = squeeze(entry(A)).';
eb = squeeze(entry(b)).';
eu = squeeze(entry(u)).';
assert(isequal(eu,eA\eb))
%% Mixed product, D2 scalar
rng default
[A,l,lp] = tensor(rand(1,1,9,9));
[u,i,~] = tensor(rand(1,1,8,9));
b(i,lp) = A(l,lp)*u(i,~l);
assert(isequal(size(b),[1 1 8 9]))
eA = squeeze(entry(A)).';
eu = squeeze(entry(u)).';
eb = squeeze(entry(b)).';
assert(isequal(eb,eA*eu))
%% Outer addition, D1 scalar
rng default
[c,j] = tensor(rand(1,1,9));
[x,i] = tensor(rand(1,1,8));
l(i,j) = log(c(j)+x(i));
ec = shiftdim(entry(c),-1);
el = log(ec+entry(x));
assert(isequal(entry(l),el))
%% Mixed product, D2 matrix
rng default
[A,i,~] = tensor(rand(9,5,7,6));
[B,~,j] = tensor(rand(5,8,7,6));
C = A(i,~j)*B(i,~j);
assert(isequal(index(C),[i ~j]))
assert(isequal(size(C),[9 8 7 6]))
eC = pagemtimes(entry(A),entry(B));
assert(isequal(entry(C),eC))
%% Trace contraction, D2 matrix
rng default
[C,i,~] = tensor(rand(9,9,8,8));
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
[C,i,~] = tensor(rand(9,9,8,8));
d = diag(C(i,i));
assert(isequal(index(d),i))
eC = entry(C);
ed = zeros(9,1,8);
for k = 1:8
    ed(:,:,k) = diag(eC(:,:,k,k));
end
assert(isequal(entry(d),ed))
%% Inner product, DN vector
rng default
dims = [9 1 8:-1:4];
[x,k] = tensor(randn(dims,'like',1j));
y = x(~k)'*x(~k);
assert(isequal(degree(y),0))
ex = entry(x);
ey = conj(ex(:)).'*ex(:);
assert(isequal(entry(y),ey))
assert(isreal(entry(y)))
%% Column concatenation, D1 vector
rng default
M = 9;
N = 8;
[c,i] = tensor(randn(M,1,M-2));
[~,~] = tensor(randn(N,1,N-2));
C = [ones(M,1) abs(c(i))];
assert(isequal(index(C),i))
eC = ones(M,2,M-2);
eC(:,2,:) = abs(entry(c));
assert(isequal(entry(C),eC))
%% Row concatenation, D1 vector
rng default
M = 9;
N = 8;
[~,~] = tensor(randn(M,1,M-2));
[b,j] = tensor(randn(N,1,N-2));
B = [b(j).'; ones(1,N)];
assert(isequal(index(B),~j))
eB = ones(2,N,N-2);
eB(1,:,:) = permute(entry(b),[2 1 3]);
assert(isequal(entry(B),eB))
%% Outer relation, D1 vector
rng default
M = 9;
N = 8;
[c,i] = tensor(randn(M,1,M-2));
[b,j] = tensor(randn(N,1,N-2));
A(i,~j) = b(j).'+abs(c(i));
ToF = A(i,~j) >= b(j).';
assert(isequal(index(ToF),[i ~j]))
assert(isequal(size(ToF),[M N M-2 N-2]))
assert(isa(entry(ToF),'logical'))
assert(all(entry(ToF),'all'))
%% Index concatenation, D2 matrix
rng default
[A,i,j] = tensor(rand(9,8,7,6));
[B,~,k] = tensor(rand(9,8,6,4));
C = cat(k,A(i,j),B(j,k));
assert(isequal(index(C),[i j k]))
assert(isequal(size(C),[9 8 7 6 5]))
eB = reshape(entry(B),[9 8 1 6 4]);
eC = cat(5,entry(A),repmat(eB,[1 1 7]));
assert(isequal(entry(C),eC))
