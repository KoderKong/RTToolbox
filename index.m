classdef index < handle
    properties (Access = protected)
        state = true;
        tilde
    end
    methods
        function [obj,varargout] = index(varargin)
            if nargin == 0
                obj.tilde = index(obj);
            elseif nargin == 1 && isempty(varargin{1})
                obj = repmat(obj,1,0);
            elseif nargin == 1 && isscalar(varargin{1}) ...
                    && isa(varargin{1},'index') ...
                    && isempty(varargin{1}.tilde)
                obj.state = false;
                obj.tilde = varargin{1};
            elseif nargin == 1 && isscalar(varargin{1}) ...
                    && isnumeric(varargin{1})
                obj = repmat(obj,1,varargin{1});
                for k = 1:numel(obj)
                    obj(k) = index;
                end
                if nargout > 1
                    varargout = cell(1,nargout-1);
                    [obj,varargout{:}] = deal(obj);
                end
            else
                error('Invalid constructor.');
            end
        end
        function varargout = deal(obj)
            num = min(nargout,numel(obj));
            if num > 0
                varargout = cell(1,num);
                varargout{1} = obj(1);
                for k = 2:num-1
                    varargout{k} = obj(k);
                end
                varargout{num} = obj(num:end);
            else
                varargout = {obj};
            end
        end
        function state = logical(obj)
            if isempty(obj)
                state = reshape(logical([]),size(obj));
            else
                state = reshape([obj.state],size(obj));
            end
        end
        function obj = true(obj)
            isfalse = ~[obj.state];
            obj(isfalse) = [obj(isfalse).tilde];
        end
        function obj = false(obj)
            istrue = [obj.state];
            obj(istrue) = [obj(istrue).tilde];
        end
        function obj = not(obj)
            if ~isempty(obj)
                obj = reshape([obj.tilde],size(obj));
            end
        end
    end
end
