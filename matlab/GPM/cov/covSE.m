function varargout = covSE(mode, par, varargin)

% covSE Squared Exponential covariance function with unit amplitude.
%
% Report number of hyperparameters
%  s = covSE (mode)
%  s = covSE (mode, par)
%  s = covSE (mode, par, hyp)
%
% Evaluation of k(x,x)
%  k = covSE (mode, par, hyp, x)
%  k = covSE (mode, par, hyp, x, [])
%
% Evaluation of diag(k(x,x))
%  k = covSE (mode, par, hyp, x, 'diag')
%
% Evaluation of k(x,z)
%  k = covSE (mode, par, hyp, x, z)
%
% Evaluation of function of derivatives dk (w.r.t. hyp and x)
%  [k, dk] = covSE (mode, par, hyp, x)
%  [k, dk] = covSE (mode, par, hyp, x, [])
%  [k, dk] = covSE (mode, par, hyp, x, 'diag')
%  [k, dk] = covSE (mode, par, hyp, x, z)
%
% Call covFunctions.m to get an explanation of outputs in each mode.
%
% The covariance function is:
%
% k(x,z) = exp(-maha(x,z)/2)
%
% where maha(x,z) is a squared Mahalanobis distance. The function takes a "mode"
% parameter, which specifies precisely the Mahalanobis distance used, see
% covMaha. The function returns either the number of hyperparameters (with less
% than 3 input arguments) or it returns a covariance matrix and (optionally) a
% derivative function.
%
% Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2016-05-23.
%
% See also covFunctions.m, cov/covMaha.m.

if nargin < 1, error('Mode cannot be empty.'); end                  % no default
if nargin < 2, par = []; end                                           % default
varargout = cell(max(1, nargout), 1);                  % allocate mem for output
if nargin < 4, varargout{1} = covMaha(mode,par); return, end

k = @(d2) exp(-d2/2); dk = @(d2,k) (-1/2)*k;         % covariance and derivative
[varargout{:}] = covMaha(mode, par, k, dk, varargin{:});