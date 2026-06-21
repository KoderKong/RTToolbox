function [sse,gsse,yQ] = qrssefg(Z,Z0,yQ,Id,vs)
[Ev,Eu,Qv,Qu,Qu0] = errfun(Z,yQ,Id,vs);
sse = full(entry(Ev(:)'*Ev(:)));
sse = sse+full(Eu(:)'*Eu(:));
if nargout > 1
    % dot = @(A,B) trace(A'*B);
    [~,j] = deal(~index(Z));
    Gvu = 2*dot(Ev*vs',Qv*Z');
    Guu = 4*dot(Eu*Qu0',Z0');
    assert(isequal(index(Gvu),index(Guu),j))
    Gvv = 2*(Ev*vs'*Qu);
    gsse = sparse(entry(Gvu(:)+Guu(:)));
    gsse = [gsse; sparse(entry(Gvv(:)))];
end

function [Ev,Eu,Qv,Qu,Qu0] = errfun(Z,yQ,Id,vs)
szZ = size(Z);
y = yQ(1:szZ(end));
ij = index(Z);
y = reshape(sparse1(y),1,1,numel(y));
Qu = Z*tensor(y,~ij(end));
szQ = size(Qu);
Qv = sparse1(yQ(numel(y)+1:end));
[h,i,~] = deal(~index(vs));
assert(isequal(i,~index(Qu)))
Qv = tensor(reshape(Qv,szQ),h);
Ev = Qv*Qu'*vs-Id;
k = index(Id);
assert(isequal(index(Ev),k))
Qu0 = entry(Qu(:,:,end));
I = speye(szQ(2));
Eu = Qu0'*Qu0-I;

function c = dot(A,B)
if isa(A,'tensor')
    A = tensor(conj(entry(A)),~index(A));
else
    A = conj(A);
end
assert(isa(B,'tensor'))
c = sum(A.*B,1:2);
