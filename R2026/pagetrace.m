function b = pagetrace(A)
sizeA = size(A);
sizeb = [1 1 sizeA(3:end)];
b = zeros(sizeb,class(A));
for k = 1:numel(b)
    b(k) = trace(A(:,:,k));
end
