function [acq] = LCB(x,hyp,inf_,mean_,cov_,lik_,xt,post,yt,opts,varargin)
% lower confidence bound
% varargin{1} could be algo data struct
% if lineBO build subspace
% beta defines the size of the confidence bound

    oldOpts.beta = 2;
    opts = getopts(oldOpts,opts);
    beta = opts.beta;
    if ~isempty(varargin{1}) && strcmp(varargin{1}.name,'lineBO')
       n = size(x,1);
       AlgoStruct = varargin{1};
       l = AlgoStruct.l;
       x_vec = AlgoStruct.x_vec;        
       if n > 1
           x_vec = repmat(x_vec,[n,1]);
       end
       x_vec(:,l) = x;
       x = x_vec;
    end
    [~,~,mu,var]=gp(hyp,inf_,mean_,cov_,lik_,xt,post,x);
    sig = sqrt(var);
    acq = (mu - beta * sig);
end
