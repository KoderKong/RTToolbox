function [a,Qu,Qv,Id] = arm2seg(x,y)
assert(isreal(x) && isreal(y) && y > 0)
syms x1 y1 x2 y2
syma = [x1+x2-x; y1+y2-y; x1^2+y1^2-1; x2^2+y2^2-1];
symw = [y2; x2; y1; x1];
a = sym2num(syma,symw);
a = tensor(sparse1(a));
a = simplify(a,'argdeg');
if nargout > 1
    u31 = x2-x1+x;
    u32 = y2-y1+y;
    u44 = 2*(y1*y-x1*x)+(x^2+y^2);
    symU = [-u31 2*y-u32 -1 1;
        1 0 0 0;
        u31 u32 1 -1;
        -u44*u31 -u44*u32 4*y^2-u44 u44];
    Qu = pagectranspose(sym2num(symU,symw));
    Qu = tensor(sparse1(Qu));
    Qu = simplify(Qu,'argdeg');
    if nargout > 2
        v43 = 2*(y2*y-x1*x)+(x^2-y^2);
        symV = [0 4*y^2 0 0;
            2*y 0 2*y 0;
            0 0 u44 1;
            2*y*u32 4*y^2*u31 v43 1]/4/y^2;
        Qv = sparse1([],[],size(Qu),0);
        Qv(:,:,:,end) = sym2num(symV,symw);
        Qv = tensor(sparse1(Qv));
        Qv = simplify(Qv,'argdeg');
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
    D = 0; % polynomialDegree error
else
    D = polynomialDegree(symA,symx);
    D = max(D(:));
end
[a,ij] = foo(symA,symx,sym(1),D);
[Mr,Mc] = size(symA);
N = numel(symx);
dims = [Mr Mc repmat(N+1,1,D)];
subs = num2cell(ij,1);
li = sub2ind(dims,subs{:});
MNpowD = prod(dims);
A = sparse(li,1,a,MNpowD,1);
if numel(A) > numel(symA)
    A = reshape(full(A),dims);
else
    A = reshape(A,Mr,Mc);
end

function [a,ij] = foo(symA,symx,sym1,D)
M = numel(symA);
a = cell(M,1);
ij = cell(M,1);
k = 0;
for col = 1:size(symA,2)
    for row = 1:size(symA,1)
        k = k+1;
        i = [row col];
        [a{k},ij{k}] = foo1(i,symA(k),symx,sym1,D);
    end
end
a = vertcat(a{:});
ij = vertcat(ij{:});

function [a,ij] = foo1(i,syma,symx,sym1,D)
[a,symt] = coeffs(syma,symx);
a = double(a(:));
numa = numel(a);
i = repmat(i,numa,1);
N = numel(symx);
j = repmat(N+1,numa,D);
for k = 1:numa
    symf = factor(symt(k),symx);
    if ~isequal(symf,sym1)
        numf = numel(symf);
        mask = logical(symf == symx);
        mask = cumsum(mask,'reverse');
        j(k,1:numf) = sum(mask);
    end
end
ij = [i j];
