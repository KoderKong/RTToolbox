function [d,i,j] = kdelta(dims,i,j,type)
if isrow(dims) && all(dims >= 0) && isequal(fix(dims),dims)
    degree = numel(dims);
    if nargin < 2
        i = index(degree);
    elseif numel(i) ~= degree
        error('Index argument must correspond to implied degree.')
    end
    if nargin < 3
        j = index(degree);
    elseif numel(j) ~= degree
        error('Index argument must correspond to implied degree.')
    end
    if nargin < 4
        type = 'double';
    end
    N = prod(dims);
    lk = find(reshape(speye(N,N),[],1));
    switch type
        case 'double'
            d = sparse1(lk,1,[1 1 dims dims],numel(lk));
        case 'logical'
            d = sparse1(lk,true,[1 1 dims dims],numel(lk));
        otherwise
            error('Specify ''double'' (default) or ''logical'' type.')
    end
    d = tensor(d,i,j);
else
    error('Size argument must be row vector of nonnegative integers.')
end
