function b = pagediag(A,varargin)
sizeA = size(A);
sizeb = sizeA(3:end);
num = prod(sizeb);
b(:,:,num) = diag(A(:,:,num),varargin{:});
for k = 1:num-1
    b(:,:,k) = diag(A(:,:,k),varargin{:});
end
[M,N,~] = size(b);
b = reshape(b,[M N sizeb]);
