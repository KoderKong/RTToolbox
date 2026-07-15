classdef (InferiorClasses = {?index,?sparse1}) tensor
    properties (Access = protected)
        entries
        indices
    end
    methods % Constructor, etc.
        function [obj,varargout] = tensor(arg,varargin)
            switch nargin
                case 0
                    obj.entries = [];
                    obj.indices = index(0);
                case 1
                    if isa(arg,'tensor')
                        obj = arg;
                    elseif ~isa(arg,'index')
                        obj.entries = arg;
                        obj.indices = index(ndims(arg)-2);
                    else
                        error('Invalid entries.')
                    end
                otherwise
                    if isa(arg,'index')
                        error('Invalid entries.')
                    elseif isa(arg,'tensor')
                        obj.entries = arg.entries;
                    else
                        obj.entries = arg;
                    end
                    indices = [varargin{:}];
                    num = numel(indices);
                    indices = reshape(indices,1,num);
                    num = num-(ndims(obj.entries)-2);
                    if isa(indices,'index') && num >= 0
                        obj.indices = indices;
                        obj = simpler(obj);
                    else
                        error('Invalid indices.')
                    end
            end
            if nargout > 1
                varargout = cell(1,nargout-1);
                [varargout{:}] = deal(obj.indices);
            end
        end
        function varargout = index(obj)
            if nargout > 1
                varargout = cell(1,nargout);
                [varargout{:}] = deal(obj.indices);
            else
                varargout = {obj.indices};
            end
        end
        function entries = entry(obj)
            entries = obj.entries;
        end
        function num = degree(obj)
            num = numel(obj.indices);
        end
        function num = ndims(obj)
            num = ndims(obj.entries);
        end
        function num = numel(obj)
            num = numel(obj.entries);
        end
        function num = length(obj)
            num = length(obj.entries);
        end
        function [szd,varargout] = size(obj,varargin)
            isindex = cellfun(@(arg) isa(arg,'index'),varargin);
            varargout = cell(1,nargout-1);
            if ~any(isindex)
                [szd,varargout{:}] = size(obj.entries,varargin{:});
            elseif all(isindex)
                k = true([varargin{:}]);
                [yes,pos] = ismember(k,true(obj.indices));
                pos(~yes) = numel(obj.indices)+1;
                [szd,varargout{:}] = size(obj.entries,pos+2);
            else
                error('All/no dimension arguments must be indices.')
            end
        end
        function num = end(obj,k,n)
            if builtin('isa',obj.entries,'sparse1')
                num = friend(obj.entries,k,n); % R2026
            else
                num = builtin('end',obj.entries,k,n);
            end
        end
    end
    methods % Simple unary
        function obj = uminus(obj)
            obj.entries = uminus(obj.entries);
        end
        function obj = uplus(obj)
            obj.entries = uplus(obj.entries);
        end
        function obj = conj(obj)
            obj.entries = conj(obj.entries);
        end
        function obj = not(obj)
            obj.entries = not(obj.entries);
        end
        function obj = abs(obj)
            obj.entries = abs(obj.entries);
        end
        function obj = log(obj)
            obj.entries = log(obj.entries);
        end
        function obj = round(obj,varargin)
            obj.entries = round(obj.entries,varargin{:});
        end
        function obj = transpose(obj)
            obj.entries = pagetranspose(obj.entries);
            obj.indices = not(obj.indices);
        end
        function obj = ctranspose(obj)
            obj.entries = pagectranspose(obj.entries);
            obj.indices = not(obj.indices);
        end
        function obj = trace(obj)
            obj.entries = pagetrace(obj.entries);
        end
        function obj = diag(obj,varargin)
            obj.entries = pagediag(obj.entries,varargin{:});
        end
    end
    methods % Additional unary
        function obj = subsref(obj,s)
            if isscalar(s) && strcmp(s.type,'()')
                isindex = cellfun(@(arg) isa(arg,'index'),s.subs);
                if all(isindex)
                    obj.indices = [s.subs{:}];
                    if numel(obj.indices) < ndims(obj.entries)-2
                        error('Insufficient number of indices.')
                    end
                    obj = simpler(obj);
                elseif numel(isindex) > 2 && all(isindex(3:end))
                    obj.indices = [s.subs{3:end}]; % R2026
                    if numel(obj.indices) < ndims(obj.entries)-2
                        error('Insufficient number of indices.')
                    end
                    obj = simpler(obj);
                    if any(isindex(1:2))
                        error('Rows/columns must not be indices.')
                    else
                        num = numel(obj.indices);
                        s.subs = [s.subs(1:2) repmat({':'},1,num)];
                        obj.entries = subsref(obj.entries,s);
                    end
                elseif any(isindex)
                    error('All/no subscripts must be indices.')
                else
                    obj.entries = subsref(obj.entries,s);
                end
            else
                obj = subsref(obj.entries,s);
            end
        end
        function obj = sum(obj,varargin)
            k = cellfun(@(arg) isa(arg,'index'),varargin);
            if any(k)
                yes = ismember(true(obj.indices),true([varargin{k}]));
                if any(yes)
                    num = numel(obj.indices)-ndims(obj.entries)+2;
                    dim = [size(obj.entries) ones(1,num)];
                    pos = find(yes)+2;
                    obj.entries = sum(obj.entries,pos,varargin{~k});
                    dim(pos) = [];
                    obj.entries = reshape(obj.entries,dim);
                    obj.indices(yes) = [];
                end
            else
                obj.entries = sum(obj.entries,varargin{:});
            end
        end
        function obj = permute(obj,varargin)
            isindex = cellfun(@(arg) isa(arg,'index'),varargin);
            if all(isindex)
                newindices = [varargin{:}];
                numnew = numel(unique(true(newindices)));
                if numel(newindices) == numnew
                    [isold,pos] = ismember(newindices,obj.indices);
                    numold = sum(isold);
                    if numel(obj.indices) == numold
                        pos(~isold) = numold+1:numnew;
                        pos = [1 2 pos+2];
                        obj.entries = permute(obj.entries,pos);
                        obj.indices = newindices;
                    else
                        error('All old indices must be retained.');
                    end
                else
                    error('New indices must be truly unique.');
                end
            elseif any(isindex)
                error('All/no arguments must be indices.');
            else
                obj.entries = permute(obj.entries,varargin{:});
            end
        end
    end
    methods % Simple binary
        function obj = plus(obj,arg)
            obj = binary(@plus,obj,arg);
        end
        function obj = minus(obj,arg)
            obj = binary(@minus,obj,arg);
        end
        function obj = eq(obj,arg)
            obj = binary(@eq,obj,arg);
        end
        function obj = ne(obj,arg)
            obj = binary(@ne,obj,arg);
        end
        function obj = lt(obj,arg)
            obj = binary(@lt,obj,arg);
        end
        function obj = gt(obj,arg)
            obj = binary(@gt,obj,arg);
        end
        function obj = le(obj,arg)
            obj = binary(@le,obj,arg);
        end
        function obj = ge(obj,arg)
            obj = binary(@ge,obj,arg);
        end
        function obj = and(obj,arg)
            obj = binary(@and,obj,arg);
        end
        function obj = or(obj,arg)
            obj = binary(@or,obj,arg);
        end
        function obj = times(obj,arg)
            obj = binary(@times,obj,arg);
        end
        function obj = ldivide(obj,arg)
            obj = binary(@ldivide,obj,arg);
        end
        function obj = rdivide(obj,arg)
            obj = binary(@rdivide,obj,arg);
        end
        function obj = power(obj,arg)
            obj = binary(@power,obj,arg);
        end
    end
    methods % Additional binary
        function obj = subsasgn(obj,s,arg)
            if isscalar(s) && strcmp(s.type,'()')
                isindex = cellfun(@(arg) isa(arg,'index'),s.subs);
                if all(isindex)
                    obj = permute(arg,[s.subs{:}]);
                    return % subsasgn
                elseif any(isindex)
                    error('All/no arguments must be indices.')
                end
            end
            if isa(arg,'tensor')
                arg = arg.entries;
            end
            obj.entries = subsasgn(obj.entries,s,arg);
            num = (ndims(obj.entries)-2)-numel(obj.indices);
            obj.indices = [obj.indices index(num)];
        end
        function obj = mtimes(obj,arg)
            obj = mbinary(@pagemtimes,obj,arg,[1 2 3],[2 1 3]);
        end
        function obj = mldivide(obj,arg)
            if isa(obj,'tensor')
                obj.indices = not(obj.indices);
            end
            obj = mbinary(@pagemldivide,obj,arg,[2 1 3],[2 1 3]);
        end
        function obj = mrdivide(obj,arg)
            if isa(arg,'tensor')
                arg.indices = not(arg.indices);
            end
            obj = mbinary(@pagemrdivide,obj,arg,[1 2 3],[1 2 3]);
        end
    end
    methods % N-ary operations
        function obj = horzcat(varargin)
            [i,k,varargin{:}] = alignn(varargin{:});
            entry = pagehorzcat(varargin{:});
            obj = tensor(entry,i);
            obj = sum(obj,k);
        end
        function obj = vertcat(varargin)
            [i,k,varargin{:}] = alignn(varargin{:});
            entry = pagevertcat(varargin{:});
            obj = tensor(entry,i);
            obj = sum(obj,k);
        end
        function obj = cat(j,varargin)
            if isscalar(j)
                [i,k,varargin{:}] = alignn(varargin{:});
                if isa(j,'index')
                    truej = true(j);
                    k(true(k) == truej) = [];
                    pos = find(true(i) == truej);
                    if isempty(pos)
                        i = [i j];
                        pos = numel(i);
                    else
                        i(pos) = j;
                    end
                    entry = pagecat(pos+2,varargin{:});
                else
                    entry = pagecat(j,varargin{:});
                end
                obj = tensor(entry,i);
                obj = sum(obj,k);
            else
                error('Dimension argument must be a scalar.')
            end
        end
        function is = isequal(varargin)
            [~,k,varargin{:}] = alignn(varargin{:});
            is = isempty(k) && isequal(varargin{:});
        end
    end
    methods (Access = protected)
        function obj = simpler(obj)
            objindices = obj.indices;
            [~,fwd,bwd] = unique(true(objindices),'stable');
            if numel(fwd) < numel(bwd)
                num = numel(obj.indices)-(ndims(obj.entries)-2);
                dim1 = [size(obj.entries) ones(1,num)];
                dim2 = dim1(3:end);
                fwd = fwd'; % Row vector
                bwd = bwd'; % Row vector
                dim2fwd = dim2(fwd);
                if isequal(dim2fwd(bwd),dim2)
                    mapb = [1 2 bwd+2];
                    dim2 = [dim1([1 2]) dim2fwd];
                    if issparse(obj.entries)
                        mapf = [1 2 fwd+2]; % R2026
                        [li,~,nz] = find(obj.entries(:));
                        [li,k] = tensor.filter(dim1,li,mapf,mapb);
                        obj.entries = sparse1(li,nz(k),dim2,numel(li));
                    else
                        li = (1:prod(dim2))';
                        li = tensor.select(li,dim2,mapb,dim1);
                        obj.entries = reshape(obj.entries(li),dim2);
                    end
                    obj.indices = objindices(fwd);
                    issum = obj.indices(bwd) ~= objindices;
                    if any(issum)
                        obj = sum(obj,objindices(issum));
                    end
                else
                    error('Attracted dimensions must be compatible.');
                end
            end
        end
        function obj = binary(fun,arg1,arg2)
            [i,k,arg1,arg2] = alignn(arg1,arg2);
            entry = fun(arg1,arg2);
            obj = tensor(entry,i);
            obj = sum(obj,k);
        end
        function [indout,indsum,varargout] = alignn(varargin)
            nargin = numel(varargin);
            indin = cell(1,nargin);
            varargout = cell(1,nargin);
            for k = 1:nargin
                if isa(varargin{k},'tensor')
                    indin{k} = index(varargin{k});
                    varargout{k} = varargin{k}.entries;
                elseif ismatrix(varargin{k})
                    indin{k} = index(0);
                    varargout{k} = varargin{k};
                else
                    error('Non-tensor argument must be 2-D array.')
                end
            end
            indout = unique([indin{:}],'stable');
            [indref,fwd,bwd] = unique(true(indout),'stable');
            indout = indout(fwd');
            bwd(fwd) = [];
            indsum = indout(bwd');
            ndims = numel(indout)+2;
            for k = 1:nargin
                [yes,pos] = ismember(indref,true(indin{k}));
                varargout{k} = permute(varargout{k},[1 2 pos(yes)+2]);
                head = size(varargout{k});
                tail = ones(1,sum(yes)-numel(head)+2);
                dims = ones(1,ndims);
                dims([true true yes]) = [head tail];
                varargout{k} = reshape(varargout{k},dims);
            end
        end
        function obj = mbinary(fun,obj1,obj2,ord1,ord2)
            ist1 = isa(obj1,'tensor');
            ist2 = isa(obj2,'tensor');
            if ist1 && ist2
                [obj1,obj2,ind,dim,seq] = mbinary2(obj1,obj2,ord1,ord2);
                obj = tensor(fun(obj1,obj2),ind);
                obj.entries = reshape(obj.entries,dim);
                obj.entries = permute(obj.entries,seq);
            elseif ist1 && ismatrix(obj2)
                obj1.entries = fun(obj1.entries,obj2);
                obj = obj1;
            elseif ismatrix(obj1) && ist2
                obj2.entries = fun(obj1,obj2.entries);
                obj = obj2;
            else
                error('Non-tensor argument must be 2-D array.');
            end
        end
        function [obj1,obj2,ind,dim,seq] = mbinary2(obj1,obj2,ord1,ord2)
            iss1 = isequal(size(obj1.entries,1:2),[1 1]);
            iss2 = isequal(size(obj2.entries,1:2),[1 1]);
            if iss1 && ~iss2
                [i,j] = index(2);
                dim2 = [1 1 size(obj2.entries)];
                obj2.entries = reshape(obj2.entries,dim2);
                obj2.indices = [i j obj2.indices];
            elseif iss2 && ~iss1
                [i,j] = index(2);
                dim1 = [1 1 size(obj1.entries)];
                obj1.entries = reshape(obj1.entries,dim1);
                obj1.indices = [i j obj1.indices];
            end
            [pos1,pos2,out1,out2,ent] = align2(obj1,obj2);
            ind = [out1 out2 ent];
            [obj1,dim1] = tensor.lattice(obj1.entries,pos1,ord1);
            [obj2,dim2] = tensor.lattice(obj2.entries,pos2,ord2);
            if isequal([dim1{2:3}],[dim2{2:3}])
                dim = dim1{1};
                num1 = numel(dim);
                dim = [dim dim2{1} dim1{3}];
                num2 = numel(dim);
                seq = [1 num1+1 2:num1 num1+2:num2];
            else
                error('Unequal inner/entrywise dimension sizes.');
            end
            if xor(iss1,iss2)
                pos = [find(ind == i) find(ind == j)];
                sqz = 1:numel(ind)+2;
                ind(pos) = [];
                pos = pos+2;
                sqz(pos) = [];
                sqz = [pos sqz(3:end) 1:2];
                seq = seq(sqz);
            end
        end
        function [pos1,pos2,out1,out2,ent] = align2(obj1,obj2)
            ind1 = true(obj1.indices);
            ind2 = true(obj2.indices);
            [yes,pos] = ismember(ind1,ind2);
            dup = pos(yes);
            yup = obj1.indices(yes) == obj2.indices(dup);
            outer = 1:numel(ind2);
            outer(dup) = [];
            inner = yes;
            inner(yes) = ~yup;
            entry = yes;
            entry(yes) = yup;
            yes = ~yes;
            pos1 = {find(yes) find(inner) find(entry)};
            pos2 = {outer pos(inner) pos(entry)};
            out1 = obj1.indices(yes);
            out2 = obj2.indices(outer);
            ent = obj1.indices(entry);
        end
    end
    methods (Access = protected, Static)
        function li1 = select(li2,dim2,bwd,dim1)
            li2 = li2-1;
            dim2 = cumprod(dim2(1:end-1));
            bwd = bwd-1;
            dim1 = [1 cumprod(dim1(1:end-1))];
            m = numel(li2);
            li1 = zeros(m,1,class(li2));
            n = numel(dim2);
            for j = n:-1:1
                tmp = rem(li2,dim2(j));
                li2 = (li2-tmp)/dim2(j);
                dim = dim1(bwd == j);
                if ~isempty(dim)
                    li1 = li1+sum(li2.*dim,2);
                end
                li2 = tmp; % Remainder
            end
            dim = dim1(bwd == 0);
            if ~isempty(dim)
                li1 = li1+sum(li2.*dim,2);
            end
            li1 = li1+1;
        end
        function [arg,dims] = lattice(arg,pos,ord)
            pos = pos(ord);
            row = [1 pos{1}+2];
            col = [2 pos{2}+2];
            tab = pos{3}+2;
            seq = [row col tab];
            num = max(seq)-ndims(arg);
            dim = [size(arg) ones(1,num)];
            row = dim(row);
            col = dim(col);
            tab = dim(tab);
            dim = [prod(row) prod(col) prod(tab)];
            arg = permute(arg,seq);
            arg = reshape(arg,dim);
            dims(ord) = {row,col,tab};
        end
    end
    methods % Release 2026 (R2026)
        function varargout = find(obj,varargin)
            varargout = cell(1,nargout);
            [varargout{:}] = find(obj.entries,varargin{:});
        end
        function disp(obj,varargin)
            disp(obj.entries,varargin{:})
        end
        function obj = sparse(obj)
            obj.entries = sparse(obj.entries);
        end
        function obj = sparse1(obj)
            obj.entries = sparse1(obj.entries);
        end
        function obj = full(obj)
            obj.entries = full(obj.entries);
        end
        function obj = double(obj)
            obj.entries = double(obj.entries);
        end
        function obj = logical(obj)
            obj.entries = logical(obj.entries);
        end
        function num = nzmax(obj)
            num = nzmax(obj.entries);
        end
        function num = nnz(obj)
            num = nnz(obj.entries);
        end
        function is = issparse(obj)
            is = issparse(obj.entries);
        end
        function is = isreal(obj)
            is = isreal(obj.entries);
        end
        function ind = subsindex(obj)
            ind = subsindex(obj.entries);
        end
        function varargout = qr(a,DeltaD,varargin)
            isf = cellfun(@(arg) isa(arg,'function_handle'),varargin);
            if sum(isf) > 1
                error('Supply no more than one function handle.')
            end
            [a,sa] = cmpa(a);
            [Astar,szQ,sQu] = makecoeff(a*sa',DeltaD);
            [vs_,sr] = cmpvs(sQu,sa);
            Z = makebasis(Astar,szQ,sQu,varargin{isf});
            y = makeinity(Z,szQ);
            [Qu,Qv,Id,vs,sQv,sId] = optimize(Z,y,sQu,varargin{~isf});
            r = Qu'*a*vs_;
            switch nargout
                case {0,1}
                    varargout = {r*sr'};
                case 2
                    varargout = {Qu*sQu,r*sr'};
                case 3
                    varargout = {Qv*sQv',Qu*sQu,r*sr'};
                case 4
                    varargout = {Qv*sQv',Qu*sQu,r*sr',Id*sId'};
                case 5
                    varargout = {Qv*sQv',Qu*sQu,r*sr',Id*sId', ...
                        vs*sQv*sQu*sId'};
            end
        end
        function [Qu,Qv,R] = qq(Qu,Qv)
            Qu0 = Qu.entries(:,:,end);
            J = sparse(Qu0'*Qu0);
            [U,flag] = chol(flip(flip(J,2),1));
            assert(isequal(flag,0))
            L = flip(flip(U,2),1);
            Qu = Qu/L;
            Qv = Qv*L';
            if nargout > 2
                R = inv(L)';
            end
        end
        function rs = simplify(r,option)
            szr = size(r,1:degree(r)+2);
            k = 3:numel(szr);
            N = size(r,3);
            if any(szr(k) ~= N)
                error('Dimension sizes of indices must all equate.')
            end
            [lk,~,nz] = find(r.entries(:));
            subs = cell(1,numel(szr));
            [subs{:}] = ind2sub(szr,lk);
            subs(k) = num2cell(sort([subs{k}],2),1);
            lk = sub2ind(szr,subs{:});
            if nargin < 2 || strcmpi(option,'mindeg')
                rs = sparse(lk,1,nz,numel(r),1);
                [lk,~,nz] = find(rs);
                [subs{:}] = ind2sub(szr,lk);
                isne = any([subs{k}] ~= N,1);
                keep = 1:find(isne,1,'last')+2;
                szr = szr(keep);
                subs = subs(keep);
                lk = sub2ind(szr,subs{:});
            elseif ~strcmpi(option,'argdeg')
                error('Use ''mindeg'' or ''argdeg'' as option.')
            end
            rs = sparse1(lk,nz,szr,numel(lk));
            k = index(r);
            k = k(1:numel(szr)-2);
            rs = tensor(rs,k);
        end
        function [x,n] = roots(r)
            dims = size(r.entries);
            Npp = size(r.entries,3);
            if isequal(dims(1:2),[1 1]) && all(dims(3:end) == Npp)
                [lk,~,nz] = find(r.entries(:));
                subs = cell(1,numel(dims)-2);
                [~,~,subs{:}] = ind2sub(dims,lk);
                subs = [subs{:}];
                n = min(subs,[],'all');
                isn = subs == n;
                if all(isn | subs == Npp,'all')
                    powx = sum(isn,2);
                    degx = max(powx,[],'all');
                    p = sparse(1,powx+1,nz,1,degx+1);
                    p = flip(p);
                    x = roots(p(find(p,1):end));
                else
                    error('Supply a polynomial that is univariate only.')
                end
            else
                error('Dimension sizes must specify one polynomial.')
            end
        end
        function f = polyval(a,varargin)
            x = vertcat(varargin{:});
            if isa(x,'tensor')
                error('Argument concatenation must not yield tensor.')
            end
            f = a.entries;
            szf = size(f);
            szx = size(x);
            if any(szf(3:end) ~= szx(1))
                error('Coefficient pages must match argument rows.')
            end
            if builtin('isa',f,'sparse1') && isa(x,'sym')
                f = full(f);
            end
            assert(isvector(x))
            for j = numel(szf):-1:3
                f = reshape(f,[],szx(1));
                f = f*x;
            end
            f = reshape(f,szf(1:2));
            if ~issparse(x)
                f = full(f);
            end
        end
    end
    methods (Access = protected) % R2026
        function [Astar,szQ,sQu,sr] = makecoeff(a,DeltaD)
            [M,one,N,~] = size(a);
            assert(one == 1)
            h = index(DeltaD);
            i = index(a);
            hi = [h i];
            k = index(numel(hi));
            [n,o] = index(2);
            NNN = repmat(N,1,numel(hi));
            if nargout < 4
                h1 = index;
            else
                [h1,k1] = index(2);
                sr = tensor.es(NNN,~k,k1);
            end
            NN = repmat(N,1,numel(h));
            sQu = tensor.es(NN,~h,h1);
            type = class(entry(a));
            if nargout < 4
                A = kdelta(M,~n,o,type)*(a*(sQu'*vsigma(NNN,~hi,k,type))) ...
                    -a*(sQu'*vsigma([M NNN],~[n hi],[o k],type));
                assert(isequal(index(A),[~n o ~h1 k]))
                A = permute(A,[~n ~h1 o k]);
            else
                A = kdelta(M,~n,o,type)*(a*(sQu'*(vsigma(NNN,~hi,k,type)*sr))) ...
                    -a*(sQu'*(vsigma([M NNN],~[n hi],[o k],type)*sr));
                assert(isequal(index(A),[~n o ~h1 k1]))
                A = permute(A,[~n ~h1 o k1]);
            end
            szQ = size(A,[1 3:4]);
            Astar = reshape(entry(A),prod(szQ),[])';
        end
        function Z = makebasis(Astar,szQ,sQu,fun)
            if nargin < 4
                [Z,q] = null(Astar);
                Z(q,:) = Z;
            else
                Z = fun(sparse(Astar));
            end
            L = size(Z,2); % rank
            Z = reshape(sparse1(Z),[szQ L]);
            kk = index(sQu);
            k = kk(end);
            u = index;
            Z = tensor(Z,~k,~u);
        end
        function y = makeinity(Z,szQ)
            M = szQ(1);
            N = prod(szQ);
            Z = reshape(entry(Z),N,[]);
            Qend = reshape(speye(M,M),[],1);
            ell = (N-M*M+1:N)';
            y = sparse(Z(ell,:)\Qend);
        end
        function [Qu,Qv,Id,vs,sQv,sId] = optimize(Z,y,sQu,varargin)
            [~,j] = deal(~index(Z));
            numy = numel(y);
            Qu = Z*tensor(reshape(sparse1(y),1,1,numy),j);
            [Qv,Id,vs,sQv,sId] = unitaryinv(Qu,sQu);
            options = optimoptions('fminunc', ...
                'Algorithm','quasi-newton', ...
                'SpecifyObjectiveGradient',true);
            options = optimoptions(options,varargin{:});
            if options.MaxIterations > 0
                assert(isreal(Z) && isreal(y))
                Z0 = Z.entries(:,:,end,:);
                M = size(Z,1);
                Z0 = tensor(reshape(Z0,M,M,numel(y)),~j);
                yQ = sparse([y; Qv.entries(:)]);
                yQ = fminunc(@(yQ) qrssefg(Z,Z0,yQ,Id,vs),yQ,options);
                y = yQ(1:numy);
                Qv.entries(:) = yQ(numy+1:end);
                Qu = Z*tensor(reshape(sparse1(y),1,1,numy),j);
            end
        end
        function [a,sa] = cmpa(a)
            a = simplify(a,'argdeg');
            dims = size(a,3:degree(a)+2);
            sa = tensor.es(dims,~index(a),index);
            a = a*sa;
        end
        function [vs,sr] = cmpvs(sQu,sa,varargin)
            szi = size(sQu,3:ndims(sQu)-1);
            i = ~index(sQu);
            i(end) = [];
            szj = size(sa,3:ndims(sa)-1);
            j = ~index(sa);
            j(end) = [];
            szk = [szi szj];
            k = index(degree(sQu)+degree(sa)-2);
            vs = vsigma(szk,~[i j],k,varargin{:});
            sr = tensor.es(szk,~k,index);
            vs = sQu'*(sa'*(vs*sr));
        end
        function [Qv,Id,vs,sQv,sId] = unitaryinv(Qu,sQu)
            M = size(Qu,1);
            N = size(sQu,3);
            DeltaD = degree(sQu)-1;
            szh = repmat(N,1,DeltaD);
            h = index(DeltaD);
            sQv = tensor.es(szh,~h,index);
            [vs,sId] = cmpvs(sQv,sQu);
            assert(isequal(degree(vs),3))
            [~,~,k] = index(vs);
            szk = size(sId,k);
            eId = sparse1([],[],[M M szk],0);
            eId(:,:,end) = speye(M,M);
            Id = tensor(eId,k);
            warning off
            Qv = Id/(Qu'*vs);
            warning on
        end
    end
    methods (Access = protected, Static) % R2026
        function [li,k] = filter(dims,li,fwd,bwd)
            subs = cell(1,numel(dims));
            [subs{:}] = ind2sub(dims,li);
            isk = true(numel(li),1);
            for j = fwd
                tmp = subs(bwd == j);
                n = numel(tmp);
                if n > 1
                    isk = isk & all(tmp{1} == [tmp{2:n}],2);
                end
            end
            dims = dims(fwd);
            subs = subs(fwd);
            k = find(isk);
            for j = 1:numel(subs)
                subs{j} = subs{j}(k);
            end
            li = sub2ind(dims,subs{:});
        end
        function [s,i,j] = es(dims,i,j)
            assert(isrow(dims) && all(diff(dims) >= 0))
            degree = numel(dims);
            assert(numel(i) == degree && isscalar(j))
            N = prod(dims);
            li = (1:N)';
            if degree > 1
                subs = cell(1,degree);
                [subs{:}] = ind2sub(dims,li);
                subs = num2cell(sort([subs{:}],2),1);
                lj = sub2ind(dims,subs{:});
                k = find(li == lj);
                dim1 = numel(k);
                s = sparse1(sparse(li(k),1:dim1,1,N,dim1));
                s = reshape(s,[1 1 dims dim1]);
            else
                s = sparse1(sparse(li,li,1,N,N));
                s = reshape(s,[1 1 dims dims]);
            end
            s = tensor(s,i,j);
        end
    end
end
