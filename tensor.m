classdef (InferiorClasses = ?index) tensor
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
                        obj = simplify(obj);
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
            num = builtin('end',obj.entries,k,n);
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
                    obj = simplify(obj);
                elseif any(isindex)
                    error('All/no arguments must be indices.')
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
        function obj = simplify(obj)
            objindices = obj.indices;
            [~,fwd,bwd] = unique(true(objindices),'stable');
            if numel(fwd) < numel(bwd)
                num = numel(obj.indices)-(ndims(obj.entries)-2);
                dim1 = uint64([size(obj.entries) ones(1,num)]);
                dim2 = dim1(3:end);
                fwd = fwd'; % Row vector
                bwd = bwd'; % Row vector
                dim2fwd = dim2(fwd);
                if isequal(dim2fwd(bwd),dim2)
                    map = [1 2 bwd+2];
                    dim2 = [dim1([1 2]) dim2fwd];
                    li = (1:prod(dim2,'native'))';
                    li = tensor.select(li,dim2,map,dim1);
                    obj.entries = reshape(obj.entries(li),dim2);
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
                    li1 = li1+sum(li2.*dim,2,'native');
                end
                li2 = tmp; % Remainder
            end
            dim = dim1(bwd == 0);
            if ~isempty(dim)
                li1 = li1+sum(li2.*dim,2,'native');
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
end
