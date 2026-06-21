function varargout = vsigma(varargin)
ist = cellfun(@(arg) isa(arg,'tensor'),varargin);
varargout = cell(1,nargout);
if all(ist)
    [varargout{:}] = vsigma1(varargin{:});
else
    [varargout{:}] = vsigma2(varargin{:});
end

function varargout = vsigma1(varargin)
dims = cellfun(@(a) size(a,index(a)),varargin,'UniformOutput',false);
dims = [dims{:}];
i = cellfun(@(arg) index(arg),varargin,'UniformOutput',false);
i = [i{:}];
numj = numel(i);
if numel(unique(true(i))) ~= numj
    error('No index or indices must repeat across the arguments.')
end
type = cellfun(@(a) class(entry(a)),varargin,'UniformOutput',false);
if any(strcmp(type,'double'))
    type = 'double';
elseif all(strcmp(type,'logical'))
    type = 'logical';
else
    error('Arguments must have ''double'' or ''logical'' entries.')
end
varargout = cell(1,nargout);
[varargout{:}] = vsigma2(dims,~i,index(numj),type);

function [s,i,j] = vsigma2(dims,i,j,type)
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
    li = (1:N)';
    if degree > 1 && all(diff(dims) >= 0)
        subs = cell(1,degree);
        [subs{:}] = ind2sub(dims,li);
        subs = num2cell(sort([subs{:}],2),1);
        lj = sub2ind(dims,subs{:});
    elseif degree > 1
        warning('Size argument is a row vector that decreases.')
        subs = cell(1,degree);
        [subs{:}] = ind2sub(dims,li);
        subs = sort([subs{:}],2);
        isgt = any(subs > dims,2);
        subs(isgt,:) = [];
        subs = num2cell(subs,1);
        lj = sub2ind(dims,subs{:});
        li(isgt) = [];
    else
        lj = li;
    end
    lk = find(reshape(sparse(li,lj,true,N,N),[],1));
    switch type
        case 'double'
            s = sparse1(lk,1,[1 1 dims dims],numel(lk));
        case 'logical'
            s = sparse1(lk,true,[1 1 dims dims],numel(lk));
        otherwise
            error('Specify ''double'' (default) or ''logical'' type.')
    end
    s = tensor(s,i,j);
else
    error('Size argument must be row vector of nonnegative integers.')
end
