function argout = pagecat(pos,varargin)
nargin = numel(varargin);
ndimsin = cellfun(@ndims,varargin);
ndimsout = max(max(ndimsin),max(pos));
dimsin = ones(nargin,ndimsout);
for k = 1:nargin
    dimsin(k,1:ndimsin(k)) = size(varargin{k});
end
dimsout = repmat(max(dimsin),nargin,1);
nrep = dimsin ~= 1;
dimsrep = dimsout;
if all(pos > 2)
    j = [1 2 pos];
else
    j = [1 2];
end
dimsout(:,j) = dimsin(:,j);
nrep(:,j) = true;
dimsrep(nrep) = 1;
for k = 1:nargin
    varargin{k} = repmat(varargin{k},dimsrep(k,:));
    varargin{k} = reshape(varargin{k},dimsout(k,:));
end
argout = cat(pos,varargin{:});
