function [hyp_new]=gpTrain(hyp, inf_, mean_, cov_, lik_, x, y, opts, varargin)
    if opts.safeOpt || opts.moSaOpt
        safeOpts = opts.safeOpts;
        I = y > safeOpts.threshold;
        y(I) = safeOpts.threshold;
    end
    oldOpts.mode = 2;
    if isfield(opts,'minFunc'), opts = opts.minFunc; else, opts = []; end
    opts=getopts(oldOpts,opts);
    mode = opts.mode;
    if mode == 1
        oldOpts.showIts = 1;
        opts = getopts(oldOpts,opts);
        if isempty(varargin)
            hyp_new = gradient_descent(hyp,@gp,opts,inf_,mean_,cov_,lik_,x,y);
        else
            hyp_new = gradient_descent(hyp,@gp,opts,inf_,mean_,cov_,lik_,x,y, varargin{:});
        end
    elseif mode == 2
        oldOpts.MaxFunEvals = 150;
        oldOpts.Method = 'qnewton';
        oldOpts.progTol = eps;
        oldOpts.optTol = eps;
        opts = getopts(oldOpts,opts);
        [hyp_new,~,~]=minimize_minfunc(hyp, @gp, opts, inf_, mean_, cov_, lik_, x, y);
    else
        oldOpts.length = -150;
        opts = getopts(oldOpts,opts);
        [hyp_new,~,~]=minimize(hyp, @gp, opts.length, inf_, mean_, cov_, lik_, x, y);
    end
end
