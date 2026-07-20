function [a,Qu,Qv,Id] = arm2seg(x,y)
assert(isreal(x) && isreal(y) && y > 0)
syms x1 y1 x2 y2
syma = [x1+x2-x; y1+y2-y; x1^2+y1^2-1; x2^2+y2^2-1];
symw = [y2; x2; y1; x1];
a = sym2num(syma,symw);
a = tensor(a);
if nargout > 1
    u31 = x2-x1+x;
    u32 = y2-y1+y;
    u44 = 2*(y1*y-x1*x)+(x^2+y^2);
    symU = [-u31 2*y-u32 -1 1;
        1 0 0 0;
        u31 u32 1 -1;
        -u44*u31 -u44*u32 4*y^2-u44 u44];
    Qu = pagectranspose(sym2num(symU,symw));
    Qu = tensor(Qu);
    if nargout > 2
        v43 = 2*(y2*y-x1*x)+(x^2-y^2);
        symV = [0 4*y^2 0 0;
            2*y 0 2*y 0;
            0 0 u44 1;
            2*y*u32 4*y^2*u31 v43 1]/4/y^2;
        Qv = sparse1([],[],size(Qu),0);
        Qv(:,:,:,end) = sym2num(symV,symw);
        Qv = tensor(Qv);
        if nargout > 3
            h = index(Qv);
            i = ~index(Qu);
            I = speye(size(Qu,2));
            Id = sparse1([],[],[size(Qv) size(Qu,~i)],nnz(I));
            Id(:,:,end) = I;
            Id = tensor(Id,h,i);
        end
    end
end

function A = sym2num(symA,symx)
if ~ismatrix(symA) || ~iscolumn(symx)
    error('Invalid dimension size(s)!')
end
if isempty(symx)
    D = 0; % polynomialDegree
else
    D = polynomialDegree(symA,symx);
    D = max(D(:));
end
[Mr,Mc] = size(symA);
N = numel(symx);
dims = [Mr Mc repmat(N+1,1,D)];
[subs,nz] = subsnz(symA,symx,D);
subs = num2cell(subs,1);
lk = sub2ind(dims,subs{:});
A = sparse1(lk,nz,dims,numel(lk));

function [subs,nz] = subsnz(symA,symx,D)
[Mr,Mc] = size(symA);
subs = cell(Mr,Mc);
nz = cell(Mr,Mc);
sym1 = sym(1);
for j = 1:Mc
    for i = 1:Mr
        [subs{i,j},nz{i,j}] = foo(symA(i,j),symx,D,[i j],sym1);
    end
end
subs = vertcat(subs{:});
nz = vertcat(nz{:});

function [subs,nz] = foo(syma,symx,D,ij,sym1)
[nz,term] = coeffs(syma,symx);
num = numel(term);
ij = repmat(ij,num,1);
N = numel(symx);
k = repmat(N+1,num,D);
for t = 1:num
    fact = factor(term(t),symx);
    if ~isequal(fact,sym1)
        mask = logical(fact == symx);
        [var,~] = find(mask);
        k(t,1:numel(fact)) = sort(var);
    end
end
subs = [ij k];
nz = double(nz(:));
