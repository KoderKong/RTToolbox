classdef sparse1
    properties (Access = protected)
        dims % Dimension sizes of a sparse multidimensional array
        data % Nonzero vector of a sparse multidimensional array
    end
    methods % Construct 1
        function obj = sparse1(varargin)
            switch nargin
                case 1
                    if builtin('isa',varargin{1},'sparse1')
                        obj = varargin{1};
                    else
                        obj.dims = size(varargin{1});
                        obj.data = sparse(varargin{1}(:));
                    end
                case 2
                    if isscalar(varargin{1}) && isscalar(varargin{2})
                        obj.dims = sparse1.chkdims([varargin{:}]);
                    else
                        error('Row and column sizes must be scalars.')
                    end
                    obj.data = sparse(prod(obj.dims),1);
                case 4
                    obj.dims = sparse1.chkdims(varargin{3});
                    obj.data = sparse(varargin{1},1,varargin{2},...
                        prod(obj.dims),1,varargin{4});
                case {3,5,6}
                    i = varargin{1}(:);
                    j = varargin{2}(:);
                    if nargin < 4
                        dims = sparse1.chkdims([max(i) max(j)]);
                    elseif isscalar(varargin{4}) && isscalar(varargin{5})
                        dims = sparse1.chkdims([varargin{4:5}]);
                    else
                        error('Row and column sizes must be scalars.')
                    end
                    lk = sub2ind(dims,i,j);
                    if nargin < 6
                        obj = sparse1(lk,varargin{3},dims,numel(lk));
                    else
                        obj = sparse1(lk,varargin{3},dims,varargin{6});
                    end
                otherwise
                    error('Constructor expects one to six arguments.')
            end
        end
        function obj = reshape(obj,varargin)
            is = cellfun(@isempty,varargin);
            if isscalar(varargin) && isvector(varargin{1})
                argin = varargin{1};
            elseif all(is | cellfun(@isscalar,varargin))
                argin = [varargin{:}];
            else
                error('Size arguments may be multiple scalars.')
            end
            total = numel(obj.data);
            known = prod(argin(:));
            switch sum(is)
                case 0
                    if total ~= known
                        error('Array dimensionality must not change.')
                    end
                case 1
                    varargin{is} = total/known;
                    argin = [varargin{:}];
                otherwise
                    error('Can have only one unknown dimension size.')
            end
            obj.dims = sparse1.chkdims(argin);
        end
        function num = ndims(obj)
            num = numel(obj.dims);
        end
        function varargout = size(obj,varargin)
            if isempty(varargin)
                if nargout > 1
                    varargout = cell(1,nargout);
                    num = numel(obj.dims);
                    if nargout < num
                        for k = 1:nargout-1
                            varargout{k} = obj.dims(k);
                        end
                        k = nargout:num;
                        varargout{nargout} = prod(obj.dims(k));
                    else
                        for k = 1:num
                            varargout{k} = obj.dims(k);
                        end
                        for k = num+1:nargout
                            varargout{k} = 1;
                        end
                    end
                else
                    varargout = {obj.dims};
                end
            else
                if isscalar(varargin) && isvector(varargin{1})
                    argin = varargin{1};
                elseif all(cellfun(@isscalar,varargin))
                    argin = [varargin{:}];
                else
                    error('Dimension arguments may be multiple scalars.')
                end
                argout = ones(1,numel(argin));
                isok = argin <= numel(obj.dims);
                argout(isok) = obj.dims(argin(isok));
                if nargout <= 1
                    varargout = {argout};
                elseif nargout == numel(argout)
                    varargout = cell(1,nargout);
                    for k = 1:nargout
                        varargout{k} = argout(k);
                    end
                else
                    error('Too many or too few multiple size arguments.')
                end
            end
        end
        function ind = end(obj,varargin)
            ind = friend(obj,varargin{:});
        end
        function num = numel(obj)
            num = numel(obj.data);
        end
        function num = length(obj)
            if isempty(obj.data)
                num = 0;
            else
                num = max(obj.dims);
            end
        end
        function is = isempty(obj)
            is = isempty(obj.data);
        end
    end
    methods (Access = ?tensor)
        function ind = friend(obj,k,n)
            if k > numel(obj.dims)
                ind = 1;
            elseif k < n
                ind = obj.dims(k);
            else
                ind = prod(obj.dims(k:end));
            end
        end
    end
    methods % Construct 2
        function varargout = find(obj,varargin)
            switch nargout
                case {0,1}
                    lk = find(obj.data,varargin{:});
                    if isrow(obj)
                        varargout = {lk'};
                    else
                        varargout = {lk};
                    end
                case 2
                    lk = find(obj.data,varargin{:});
                    if isrow(obj)
                        [i,j] = ind2sub(obj.dims,lk');
                    else
                        [i,j] = ind2sub(obj.dims,lk);
                    end
                    varargout = {i,j};
                otherwise
                    [lk,~,nz] = find(obj.data,varargin{:});
                    if isrow(obj)
                        [varargout{1:nargout-1}] = ind2sub(obj.dims,lk');
                        varargout{nargout} = nz.';
                    else
                        [varargout{1:nargout-1}] = ind2sub(obj.dims,lk);
                        varargout{nargout} = nz;
                    end
            end
        end
        function disp(obj,arg)
            [lk,~,nz] = find(obj.data);
            subs = cell(1,numel(obj.dims));
            [subs{:}] = ind2sub(obj.dims,lk);
            subs = [subs{:}];
            if isempty(subs)
                format = sprintf('%%d%c',0xD7);
                dimstr = sprintf(format,obj.dims);
                fprintf('\tAll zero sparse:\t')
                disp(dimstr(1:end-1))
            else
                if nargin < 2 || strcmpi(arg,'sortrows')
                    [~,kseq] = sortrows(subs);
                    kseq = reshape(kseq,1,numel(lk));
                elseif strcmpi(arg,'native')
                    kseq = 1:numel(lk);
                else
                    error('Use ''sortrows'' (default) or ''native''.')
                end
                digits = floor(log10(obj.dims))+1;
                format = sprintf('%%%dd,',digits);
                format = ['\t(' format(1:end-1) ')\t'];
                for k = kseq
                    fprintf(format,subs(k,:))
                    disp(nz(k))
                end
            end
        end
        function str = class(obj)
            str = class(obj.data);
        end
        function arg = sparse(obj)
            arg = reshape(obj.data,obj.dims);
        end
        function arg = full(obj)
            arg = reshape(full(obj.data),obj.dims);
        end
        function num = nzmax(obj)
            num = nzmax(obj.data);
        end
        function num = nnz(obj)
            num = nnz(obj.data);
        end
        function is = issparse(obj)
            is = issparse(obj.data);
        end
        function is = isreal(obj)
            is = isreal(obj.data);
        end
        function is = isa(obj,arg)
            is = isa(obj.data,arg);
        end
        function obj = logical(obj)
            if islogical(obj.data)
                if isscalar(obj.data)
                    obj = obj.data;
                end
            else
                obj.data = logical(obj.data);
            end
        end
        function obj = double(obj)
            obj.data = double(obj.data);
        end
    end
    methods % Unary 1
        function obj = subsref(obj,s)
            if isscalar(s) && strcmp(s.type,'()')
                [maxi,endi,s.subs{:}] = ...
                    sparse1.subsin(obj.dims,s.subs{:});
                if all(maxi <= endi)
                    if isscalar(endi)
                        obj = subsref1(obj,s.subs{1},endi);
                    else
                        obj.dims = endi;
                        obj = subsrefn(obj,s.subs);
                    end
                else
                    error('Index exceeds dimension size.')
                end
            else
                error('Invalid indexing expression.')
            end
        end
        function ind = subsindex(obj)
            if islogical(obj.data)
                ind = find(obj.data);
            else
                error('Cast to logical before indexing.')
            end
        end
        function obj = permute(obj,seq)
            numd = numel(obj.dims);
            nums = numel(seq);
            if numd <= nums && isequal(sort(seq),1:nums)
                dims1 = [obj.dims ones(1,nums-numd)];
                dims2 = dims1(seq);
                subs = cell(1,nums);
                [lk,~,nz] = find(obj.data);
                [subs{:}] = ind2sub(dims1,lk);
                lk = sub2ind(dims2,subs{seq});
                obj.data = sparse(lk,1,nz,numel(obj.data),1);
                obj.dims = sparse1.chkdims(dims2);
            else
                error('Ensure all dimensions permute one-to-one.')
            end
        end
        function obj = sum(obj,varargin)
            obj = sumlike(@sum,obj,varargin{:});
        end
        function obj = max(obj,varargin)
            fun = @(arg,varargin) max(arg,[],varargin{:});
            obj = sumlike(fun,obj,varargin{:});
        end
        function obj = min(obj,varargin)
            fun = @(arg,varargin) min(arg,[],varargin{:});
            obj = sumlike(fun,obj,varargin{:});
        end
        function obj = all(obj,varargin)
            obj = sumlike(@all,obj,varargin{:});
        end
        function obj = any(obj,varargin)
            obj = sumlike(@any,obj,varargin{:});
        end
    end
    methods % Unary 2
        function obj = uminus(obj)
            obj.data = uminus(obj.data);
        end
        function obj = uplus(obj)
            obj.data = uplus(obj.data);
        end
        function obj = not(obj)
            obj.data = not(obj.data);
        end
        function obj = conj(obj)
            obj.data = conj(obj.data);
        end
        function obj = abs(obj)
            obj.data = abs(obj.data);
        end
        function obj = sqrt(obj)
            obj.data = sqrt(obj.data);
        end
        function obj = real(obj)
            obj.data = real(obj.data);
        end
        function obj = imag(obj)
            obj.data = imag(obj.data);
        end
        function obj = log(obj)
            obj.data = log(obj.data);
        end
        function obj = round(obj,varargin)
            obj.data = round(obj.data,varargin{:});
        end
    end
    methods % Unary 3
        function obj = transpose(obj)
            if numel(obj.dims) > 2
                error('Use pagetranspose to transpose an ND array.')
            else
                obj = permute(obj,[2 1]);
            end
        end
        function obj = ctranspose(obj)
            if numel(obj.dims) > 2
                error('Use pagectranspose to ctranspose an ND array.')
            else
                obj = permute(obj,[2 1]);
                obj.data = conj(obj.data);
            end
        end
        function obj = trace(obj)
            if numel(obj.dims) > 2
                error('Use pagetrace for the trace of an ND array.')
            else
                obj = pagetrace(obj);
            end
        end
        function obj = diag(obj,varargin)
            if numel(obj.dims) > 2
                error('Use pagediag for the diag of an ND array.')
            else
                obj = pagediag(obj,varargin{:});
            end
        end
        function obj = pagetranspose(obj)
            seq = [2 1 3:numel(obj.dims)];
            obj = permute(obj,seq);
        end
        function obj = pagectranspose(obj)
            seq = [2 1 3:numel(obj.dims)];
            obj = permute(obj,seq);
            obj.data = conj(obj.data);
        end
        function obj = pagetrace(obj)
            M = obj.dims(1);
            N = obj.dims(2);
            if M == N
                P = prod(obj.dims(3:end));
                [lk,~,nz] = find(obj.data);
                [i,j,k] = ind2sub([M N P],lk);
                iseq = i == j;
                lk = sub2ind([M P],i(iseq),k(iseq));
                obj.data = sparse(lk,1,nz(iseq),M*P,1);
                obj.dims(2) = 1;
                obj = sumlike(@sum,obj,1);
            else
                error('Matrix pages must be square.')
            end
        end
        function obj = pagediag(obj,arg)
            if nargin < 2
                arg = 0;
            end
            M = obj.dims(1);
            N = obj.dims(2);
            if M == 1 || N == 1
                error('Diagonal matrix from vector unsupported.')
            elseif isscalar(arg)
                P = prod(obj.dims(3:end));
                [lk,~,nz] = find(obj.data);
                [i,j,k] = ind2sub([M N P],lk);
                iseq = i+arg == j;
                if arg >= 0 && N >= arg
                    L = min(M,N-arg);
                    h = i(iseq);
                elseif arg < 0 && M >= -arg
                    L = min(M+arg,N);
                    h = j(iseq);
                else
                    L = 0;
                    h = zeros(0,1);
                end
                lk = sub2ind([L P],h,k(iseq));
                obj.data = sparse(lk,1,nz(iseq),L*P,1);
                obj.dims(1:2) = [L 1];
            else
                error('Specify which diagonal using a scalar.')
            end
        end
        function [Z,q] = null(obj,varargin)
            [R,p,q] = rref(obj,varargin{:});
            M = numel(p);
            N = size(R,2);
            j = 1:N;
            j(p) = [];
            R = R(1:M,j);
            I = speye(N-M);
            Z([p j],:) = [-R; I];
        end
        function [R,p,q] = rref(obj,tol)
            if nargin < 2 || isempty(tol)
                tol = eps(norm(obj.data,Inf));
                tol = max(obj.dims(1:2))*tol;
            end
            try
                assert(all(isfinite(obj.data)))
                [~,R,~,q] = spmex1('rrlu',sparse(obj));
                [M,N] = size(R);
                p = 1:M;
                [li,lj,nz] = find(R);
                iseq = li == lj;
                if tol > 0
                    pos = find(iseq & abs(nz) < tol,1);
                else
                    pos = zeros(0,1);
                end
                if isscalar(pos)
                    k = li(pos):M;
                else
                    pos = find(iseq,1,'last');
                    if isscalar(pos)
                        k = li(pos)+1:M;
                    else
                        k = min(M,N)+1:M;
                    end
                end
                if nnz(R(k,:)) > 0
                    R(k,:) = 0;
                end
                p(k) = [];
                i = 1:numel(p);
                R(i,:) = R(i,i)\R(i,:);
            catch
                [R,p] = rref(sparse(obj),tol);
                q = 1:obj.dims(2);
            end
        end
    end
    methods % N-ary
        function obj = horzcat(varargin)
            obj = cat(2,varargin{:});
        end
        function obj = vertcat(varargin)
            obj = cat(1,varargin{:});
        end
        function obj = cat(pos,varargin)
            if isscalar(pos)
                maxnd = max(cellfun(@ndims,varargin));
                other = [1:pos-1 pos+1:maxnd];
                dims1 = cellfun(@(arg) size(arg,other),varargin,...
                    'UniformOutput',false);
            else
                error('Dimension argument must be a scalar.')
            end
            if isequal(dims1{:})
                dims1 = dims1{1};
                dims2 = sum(cellfun(@(arg) size(arg,pos),varargin));
                seq = [other pos];
                for k = 1:numel(varargin)
                    varargin{k} = permute(varargin{k},seq);
                    varargin{k} = sparse1.vector(varargin{k});
                end
                obj = sparse1(vertcat(varargin{:}));
                obj = reshape(obj,[dims1 dims2]);
                seq([other pos]) = 1:numel(seq);
                obj = permute(obj,seq);
            else
                error('Dimension sizes must be consistent.');
            end
        end
        function is = isequal(varargin)
            dimsin = cellfun(@size,varargin,'UniformOutput',false);
            if isequal(dimsin{:})
                for k = 1:numel(varargin)
                    varargin{k} = sparse1.vector(varargin{k});
                end
                is = isequal(varargin{:});
            else
                is = false;
            end
        end
    end
    methods % Binary 1 and 2
        function obj = subsasgn(obj,s,rhs)
            if isscalar(s) && strcmp(s.type,'()')
                obj = sparse1(obj);
                [maxi,endi,s.subs{:}] = ...
                    sparse1.subsin(obj.dims,s.subs{:});
                [obj,endi] = resize(obj,maxi,endi);
                if isscalar(endi)
                    obj = subsasgn1(obj,s.subs{1},rhs);
                else
                    sz = obj.dims;
                    obj.dims = endi;
                    obj = subsasgnn(obj,s.subs,rhs);
                    obj.dims = sz;
                end
            else
                error('Invalid indexing expression.')
            end
        end
        function obj = plus(obj1,obj2)
            obj = binary(@plus,obj1,obj2);
        end
        function obj = minus(obj1,obj2)
            obj = binary(@minus,obj1,obj2);
        end
        function obj = eq(obj1,obj2)
            obj = binary(@eq,obj1,obj2);
        end
        function obj = ne(obj1,obj2)
            obj = binary(@ne,obj1,obj2);
        end
        function obj = lt(obj1,obj2)
            obj = binary(@lt,obj1,obj2);
        end
        function obj = gt(obj1,obj2)
            obj = binary(@gt,obj1,obj2);
        end
        function obj = le(obj1,obj2)
            obj = binary(@le,obj1,obj2);
        end
        function obj = ge(obj1,obj2)
            obj = binary(@ge,obj1,obj2);
        end
        function obj = and(obj1,obj2)
            obj = binary(@and,obj1,obj2);
        end
        function obj = or(obj1,obj2)
            obj = binary(@or,obj1,obj2);
        end
        function obj = times(obj1,obj2)
            obj = binary(@times,obj1,obj2);
        end
        function obj = rdivide(obj1,obj2)
            obj = binary(@rdivide,obj1,obj2);
        end
        function obj = ldivide(obj1,obj2)
            obj = binary(@ldivide,obj1,obj2);
        end
        function obj = power(obj1,obj2)
            obj = binary(@power,obj1,obj2);
        end
        function obj = complex(obj,arg)
            if nargin > 1
                obj = binary(@complex,obj,arg);
            else
                obj.data = complex(obj.data);
            end
        end
    end
    methods % Binary 3
        function obj = mtimes(obj1,obj2)
            dims1 = size(obj1);
            dims2 = size(obj2);
            if isequal(dims1,[1 1])
                obj = easybin(@times,obj1,obj2,dims2);
            elseif isequal(dims2,[1 1])
                obj = easybin(@times,obj1,obj2,dims1);
            elseif numel(dims1) > 2 || numel(dims2) > 2
                error('Use pagemtimes for an ND array mtimes.')
            elseif dims1(2) == dims2(1)
                [i1,k1,nz1] = find(obj1);
                [k2,j1,nz2] = find(obj2);
                [numk,cmpk] = sparse1.compind([k1(:); k2(:)]);
                [numi,cmpi,dcmpi] = sparse1.compind(i1);
                [numj,cmpj,dcmpj] = sparse1.compind(j1);
                arg1 = sparse(cmpi(i1),cmpk(k1),nz1,numi,numk);
                arg2 = sparse(cmpk(k2),cmpj(j1),nz2,numk,numj);
                [i2,j2,nz] = find(arg1*arg2); % mtimes
                obj = sparse1(dcmpi(i2),dcmpj(j2),nz,dims1(1),dims2(2));
            else
                error('Incompatible dimension sizes for mtimes.')
            end
        end
        function obj = mrdivide(obj1,obj2)
            dims1 = size(obj1);
            dims2 = size(obj2);
            if isequal(dims2,[1 1])
                obj = easybin(@rdivide,obj1,obj2,dims1);
            elseif numel(dims1) > 2 || numel(dims2) > 2
                error('Use pagemrdivide for an ND array mrdivide.')
            elseif dims1(2) == dims2(2)
                [i1,k1,nz1] = find(obj1);
                [j1,k2,nz2] = find(obj2);
                [numk,cmpk] = sparse1.compind([k1(:); k2(:)]);
                [numi,cmpi,dcmpi] = sparse1.compind(i1);
                [numj,cmpj,dcmpj] = sparse1.compind(j1);
                arg1 = sparse(cmpi(i1),cmpk(k1),nz1,numi,numk);
                arg2 = sparse(cmpj(j1),cmpk(k2),nz2,numj,numk);
                [i2,j2,nz] = find(arg1/arg2); % mrdivide
                obj = sparse1(dcmpi(i2),dcmpj(j2),nz,dims1(1),dims2(1));
            else
                error('Incompatible dimension sizes for mrdivide.')
            end
        end
        function obj = mldivide(obj1,obj2)
            dims1 = size(obj1);
            dims2 = size(obj2);
            if isequal(dims1,[1 1])
                obj = easybin(@ldivide,obj1,obj2,dims2);
            elseif numel(dims1) > 2 || numel(dims2) > 2
                error('Use pagemldivide for an ND array mldivide.')
            elseif dims1(1) == dims2(1)
                [k1,i1,nz1] = find(obj1);
                [k2,j1,nz2] = find(obj2);
                [numk,cmpk] = sparse1.compind([k1(:); k2(:)]);
                [numi,cmpi,dcmpi] = sparse1.compind(i1);
                [numj,cmpj,dcmpj] = sparse1.compind(j1);
                arg1 = sparse(cmpk(k1),cmpi(i1),nz1,numk,numi);
                arg2 = sparse(cmpk(k2),cmpj(j1),nz2,numk,numj);
                [i2,j2,nz] = find(arg1\arg2); % mldivide
                obj = sparse1(dcmpi(i2),dcmpj(j2),nz,dims1(2),dims2(2));
            else
                error('Incompatible dimension sizes for mldivide.')
            end
        end
        function obj = pagemtimes(obj1,obj2)
            obj = pagembinary(@mtimes,obj1,obj2,@times,true,true);
        end
        function obj = pagemrdivide(obj1,obj2)
            obj = pagembinary(@mrdivide,obj1,obj2,@rdivide,false,true);
        end
        function obj = pagemldivide(obj1,obj2)
            obj = pagembinary(@mldivide,obj1,obj2,@ldivide,true,false);
        end
    end
    methods (Access = protected)
        function obj = subsref1(obj,sub1,pdims)
            if strcmp(sub1,':')
                obj.dims = [pdims 1];
            else
                if isscalar(obj)
                    obj.dims = size(sub1);
                elseif iscolumn(obj)
                    obj.dims = [numel(sub1) 1];
                elseif isrow(obj)
                    obj.dims = [1 numel(sub1)];
                else
                    obj.dims = size(sub1);
                end
                obj.data = obj.data(sub1);
            end
        end
        function obj = subsrefn(obj,subs)
            iseq = cellfun(@(sub) strcmp(sub,':'),subs);
            other = 1:numel(subs);
            colon = other(iseq);
            other(colon) = [];
            dimsi = obj.dims(other);
            dimsj = obj.dims(colon);
            obj = permute(obj,[other colon]);
            [lk,~,nz] = find(obj.data);
            endi = prod(dimsi);
            endj = prod(dimsj);
            [i,j] = ind2sub([endi endj],lk);
            [numj,cmpj,dcmpj] = sparse1.compind(j);
            obj.data = sparse(i,cmpj(j),nz,endi,numj);
            [lk,dimsk] = sparse1.subsout(dimsi,subs{other});
            [i,j,nz] = find(obj.data(lk,:));
            endk = prod(dimsk);
            lk = sub2ind([endk endj],i(:),dcmpj(j(:)));
            obj.data = sparse(lk,1,nz,endk*endj,1);
            obj.dims = [dimsk dimsj];
            seq([other colon]) = 1:numel(subs);
            obj = permute(obj,seq);
        end
        function obj = sumlike(fun,obj,varargin)
            numd = numel(obj.dims);
            other = 1:numd;
            if nargin > 2 && isnumeric(varargin{1})
                pos = sort(varargin{1}(:))';
                if isempty(pos)
                    warning('Empty dimensions specify null operation.')
                elseif any(diff(pos) == 0)
                    error('Specify unique dimensions for operation.')
                end
                varargin(1) = [];
                pos(pos > numd) = [];
                other(pos) = [];
            elseif nargin > 2 && strcmpi(varargin{1},'all')
                varargin(1) = [];
                pos = other;
                other = [];
            else
                pos = find(obj.dims ~= 1,1);
                other(pos) = [];
            end
            if isempty(other)
                obj = fun(obj.data,1,varargin{:});
            else
                dimsi = obj.dims(pos);
                dimsj = obj.dims(other);
                obj = permute(obj,[pos other]);
                endi = prod(dimsi);
                endj = prod(dimsj);
                [lk,~,nz] = find(obj.data);
                [i,j] = ind2sub([endi endj],lk);
                [numj,cmpj,dcmpj] = sparse1.compind(j);
                obj.data = sparse(i,cmpj(j),nz,endi,numj);
                obj.data = fun(obj.data,1,varargin{:});
                [i,j,nz] = find(obj.data);
                obj.data = sparse(dcmpj(j),i,nz,endj,1);
                sz([pos other]) = [ones(1,numel(dimsi)) dimsj];
                obj.dims = sparse1.chkdims(sz);
            end
        end
        function [obj,endi] = resize(obj,maxi,endi)
            isgt = maxi > endi;
            if any(isgt)
                numd = numel(obj.dims);
                nume = numel(endi);
                if isgt(nume) && nume < numd
                    error('Attempt to grow along ambiguous dimension.')
                end
                [lk,~,nz] = find(obj.data);
                subs = cell(1,nume);
                [subs{:}] = ind2sub(endi,lk);
                endi(isgt) = maxi(isgt);
                lk = sub2ind(endi,subs{:});
                obj.data = sparse(lk,1,nz,prod(endi),1);
                if find(isgt,1,'last') <= numd
                    obj.dims(isgt) = maxi(isgt);
                else
                    obj.dims = sparse1.chkdims(endi);
                end
            end
        end
        function obj = subsasgn1(obj,sub1,rhs)
            if strcmp(sub1,':')
                obj.data(:) = sparse(rhs(:));
            else
                obj.data(sub1) = sparse(rhs(:));
            end
        end
        function obj = subsasgnn(obj,subs,rhs)
            iseq = cellfun(@(sub) strcmp(sub,':'),subs);
            other = 1:numel(subs);
            colon = other(iseq);
            other(colon) = [];
            dimsi = obj.dims(colon);
            dimsj = obj.dims(other);
            seq = [colon other];
            obj = permute(obj,seq);
            [lk,~,nz] = find(obj.data);
            endi = prod(dimsi);
            endj = prod(dimsj);
            [i,j] = ind2sub([endi endj],lk);
            [k,dimsk] = sparse1.subsout(dimsj,subs{other});
            [numj,cmpj,dcmpj] = sparse1.compind([j; k]);
            obj.data = sparse(i,cmpj(j),nz,endi,numj);
            if isscalar(rhs)
                obj.data(:,cmpj(k)) = full(rhs);
            else
                sz(seq) = [dimsi dimsk];
                szr = size(rhs);
                if isequal(sz(sz ~= 1),szr(szr ~= 1))
                    rhs = permute(reshape(rhs,sz),seq);
                    rhs = reshape(rhs,endi,numel(k));
                else
                    error('Incompatible dimension sizes.')
                end
                obj.data(:,cmpj(k)) = sparse(rhs);
            end
            [i,j,nz] = find(obj.data);
            lk = sub2ind([endi endj],i(:),dcmpj(j(:)));
            obj.data = sparse(lk,1,nz,endi*endj,1);
            seq(seq) = 1:numel(subs);
            obj = permute(obj,seq);
        end
        function obj = binary(fun,obj1,obj2)
            dims1 = size(obj1);
            dims2 = size(obj2);
            if isequal(dims1,[1 1]) || isequal(dims1,dims2)
                obj = easybin(fun,obj1,obj2,dims2);
            elseif isequal(dims2,[1 1])
                obj = easybin(fun,obj1,obj2,dims1);
            else
                obj = hardbin(fun,obj1,obj2,dims1,dims2);
            end
        end
        function obj = easybin(fun,obj1,obj2,dimso)
            arg1 = sparse1.vector(obj1);
            arg2 = sparse1.vector(obj2);
            obj = sparse1(fun(arg1,arg2));
            obj = reshape(obj,dimso);
        end
        function obj = hardbin(fun,obj1,obj2,dims1,dims2)
            ndims1 = numel(dims1);
            ndims2 = numel(dims2);
            maxnd = max(ndims1,ndims2);
            dims1 = [dims1 ones(1,maxnd-ndims1)];
            dims2 = [dims2 ones(1,maxnd-ndims2)];
            dimso = max(dims1,dims2);
            isne1 = dims1 ~= dimso;
            isne2 = dims2 ~= dimso;
            if all(dims1(isne1) == 1) && all(dims2(isne2) == 1)
                isk = dims1 == dims2;
                dimsk = dimso(isk);
                P = prod(dimsk);
                [i,k1,nz1,posi,dimsi,M] = ...
                    sparse1.foo(obj1,dims1,isk | dims1 == 1);
                [j,k2,nz2,posj,dimsj,N] = ...
                    sparse1.foo(obj2,dims2,isk | dims2 == 1);
                [numk,cmpk,dcmpk] = sparse1.compind([k1; k2]);
                try
                    assert(all(isfinite(nz1)) && all(isfinite(nz2)))
                    arg1 = sparse(i,cmpk(k1),nz1,M,numk);
                    arg2 = sparse(j,cmpk(k2),nz2,N,numk);
                    arg = spmex1('bsx',char(fun),arg1,arg2);
                    [ij,k,nz] = find(arg);
                    obj = sparse1(ij,dcmpk(k),nz,M*N,P);
                catch
                    MN = M*N;
                    arg1 = sparse(i,cmpk(k1),1:numel(nz1),M,numk);
                    [ij,k,pos] = find(repmat(arg1,N,1));
                    obj1 = sparse1(ij,dcmpk(k),nz1(pos),MN,P);
                    arg2 = sparse(j,cmpk(k2),1:numel(nz2),N,numk);
                    [ij,k,pos] = find(repelem(arg2,M,1));
                    obj2 = sparse1(ij,dcmpk(k),nz2(pos),MN,P);
                    obj = easybin(fun,obj1,obj2,[MN P]);
                end
                obj = reshape(obj,[dimsi dimsj dimsk]);
                seq([posi posj find(isk)]) = 1:maxnd;
                obj = permute(obj,seq);
            else
                error('Ensure dimension sizes are compatible.')
            end
        end
        function obj = pagembinary(fun,obj1,obj2,sfun,is1a,is2a)
            dims1 = size(obj1);
            dims2 = size(obj2);
            dims1a = dims1(1:2);
            dims1b = dims1(3:end);
            dims2a = dims2(1:2);
            dims2b = dims2(3:end);
            if is1a && isequal(dims1a,[1 1]) ...
                    || is2a && isequal(dims2a,[1 1])
                obj = binary(sfun,obj1,obj2);
            elseif isempty(dims1b) || isempty(dims2b) ...
                    || isequal(dims1b,dims2b)
                pages = max(prod(dims1b),prod(dims2b));
                obj1 = sparse1.blkdiag(obj1,[dims1a pages],dims1b);
                obj2 = sparse1.blkdiag(obj2,[dims2a pages],dims2b);
                obj = fun(obj1,obj2);
                [rows,cols] = size(obj);
                rows = round(rows/pages);
                cols = round(cols/pages);
                dims3 = [rows pages cols pages];
                obj = reshape(obj,dims3);
                [i,k1,j,k2,nz] = find(obj);
                assert(isequal(k1,k2)) % blkdiag
                dims3 = [rows cols pages];
                lk = sub2ind(dims3,i,j,k1);
                obj = sparse1(lk,nz,dims3,numel(lk));
                if isempty(dims2b)
                    dims3 = [rows cols dims1b];
                else
                    dims3 = [rows cols dims2b];
                end
                obj = reshape(obj,dims3);
            else
                error('Incompatible dimension sizes for operation.')
            end
        end
    end
    methods (Access = protected, Static)
        function dims = chkdims(dims)
            if isrow(dims)
                k = find(dims ~= 1,1,'last');
                if isempty(k) || k < 2
                    dims = dims(1:2);
                else
                    dims = dims(1:k);
                end
                if any(dims < 0 | fix(dims) ~= dims | isinf(dims))
                    error('Nonnegative integer dimension sizes only.')
                end
            else
                error('Specify dimension sizes as a row vector.')
            end
        end
        function [maxi,endi,varargout] = subsin(dims,varargin)
            numd = numel(dims);
            numv = numel(varargin);
            maxi = zeros(1,numv);
            if numd > numv
                endi = [dims(1:numv-1) prod(dims(numv:numd))];
            else
                endi = [dims ones(1,numv-numd)];
            end
            varargout = cell(1,numv);
            for k = 1:numv
                arg = varargin{k};
                if isobject(arg)
                    arg = subsindex(arg);
                end
                if strcmp(arg,':')
                    maxi(k) = endi(k);
                else
                    if islogical(arg)
                        arg = find(arg(:));
                    else
                        arg = double(arg(:));
                        if any(arg <= 0 | fix(arg) ~= arg)
                            error('Index not a positive integer.')
                        end
                    end
                    if isempty(arg)
                        maxi(k) = 0;
                    else
                        maxi(k) = max(arg);
                    end
                end
                varargout{k} = arg;
            end
        end
        function [numk,lk2k,k2lk] = compind(lk)
            if isempty(lk)
                k2lk = zeros(0,1);
                numk = 0;
                lk2k = sparse(0,1);
            else
                lk = sort(lk(:));
                k2lk = lk([true; diff(lk) ~= 0]);
                numk = numel(k2lk);
                lk2k = sparse(k2lk,1,1:numk);
            end
        end
        function [lk,endk] = subsout(endi,varargin)
            numv = numel(varargin);
            arg = varargin{numv};
            endk(numv) = numel(arg);
            lk = arg-1;
            for k = numv-1:-1:1
                arg = varargin{k};
                endk(k) = numel(arg);
                lk = endi(k)*lk'+(arg-1);
                lk = lk(:);
            end
            lk = lk+1;
        end
        function arg = vector(arg)
            obj = sparse1(arg);
            arg = obj.data;
        end
        function [i,j,nz,posi,dimsi,M] = foo(arg,dims,isj)
            posi = find(~isj);
            posj = find(isj);
            arg = permute(arg,[posi posj]);
            dimsi = dims(posi);
            dimsj = dims(posj);
            M = prod(dimsi);
            N = prod(dimsj);
            obj = sparse1(arg);
            [lk,~,nz] = find(obj.data);
            [i,j] = ind2sub([M N],lk);
        end
        function obj = blkdiag(arg,dimsa,dimsb)
            arg = sparse1.vector(arg);
            if isempty(dimsb) && dimsa(3) ~= 1
                arg = repmat(arg,dimsa(3),1);
            end
            [lk,~,nz] = find(arg);
            [i,j,k] = ind2sub(dimsa,lk);
            dimsa = dimsa([1 3 2 3]);
            lk = sub2ind(dimsa,i,k,j,k);
            obj = sparse1(lk,nz,dimsa,numel(lk));
            rows = dimsa(1)*dimsa(2);
            cols = dimsa(3)*dimsa(4);
            obj = reshape(obj,rows,cols);
        end
    end
end
