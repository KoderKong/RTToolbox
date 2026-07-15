%% Test1
k = index;
assert(isscalar(k))
assert(k ~= ~k)
assert(k == ~~k)
assert(true(~k) == k)
assert(false(k) == ~k)
%% Test2
k = index(2);
[i,j] = deal(k);
assert(isscalar(i))
assert(isscalar(j))
assert(all(k == [i j]))
%% Test3
[i,j] = index(2);
assert(isscalar(i))
assert(isscalar(j))
%% Test4
sz = [100 100 10];
[A,k] = tensor(rand(sz));
assert(isa(k,'index'))
%% Test5
k = [~index index(2)];
var = logical(k);
ftt = [false true true];
assert(isequal(var,ftt))
