function [x_transf]=forwardCoordTransf(cond,x,varargin)
    sort_abs = 0;
    if nargin >= 3, sort_abs = varargin{1}; end
    if nargin >=4, interval = varargin{2}; else, interval =[-1,1]; end
    if nargin >= 5, error("two man input arguments"); end
    if size(cond,1) ~= size(x,2), error("dimension of parameters and boundaries does not match"); end
    
    b = interval(2);
    a = interval(1);
    di = b-a;
    if sort_abs
        cond = sort(cond,2,"ascend","ComparisonMethod","abs");
    end
    slope_ = (cond(:,2)'-cond(:,1)')/di;
    x_transf = (x-cond(:,2)')./slope_+b;
end

