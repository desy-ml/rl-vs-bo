cond_r=[
    -1, 1;
    -1, 1;
    -1, 1;
    -1, 1;
    -1, 1;
    ];
scale=1;

cond = repmat([-1,1],size(cond_r,1),1);


cov_matern = @(varargin)covMaternard(3,varargin);

inf_ = {@infGaussLik};
mean_ = {@meanConst};
lik_ = {@likGauss};
cov_ = {@(varargin)covMaternard(3,varargin{:})};
hyp.lik = log(0.01);
hyp.mean = 10;
hyp.cov = log([0.3;0.3;0.3;0.3;0.5;10]);
acq = {@EI};
x0 = [-0.7222, 0.7222, 0.0, -0.7222, 0.0];
opts.plot=0;

opts.minFunc.mode=2;
opts.acqFunc.xi = 0.0;
opts.acqFunc.beta = 2;

opts.trainGP.acqVal = 2;%0.055;%0.5 %1D       %%% 0.05 D=1 with EI; 0.5 D = 1
opts.maxProb = 0;
opts.termCondAcq = 0.1;%0.05;%0.25;%0.5 %1D    %%% 0.05 D=1 with EI; 0.25 or 0.5 D=1; 0.2 D=2 with EI sf = 5
opts.maxIt = 150;
opts.trainGP.It = 10000;
opts.trainGP.train = 0;
opts.safeOpt = 1;
opts.moSaOpt = 1;
opts.safeOpts.threshold = 10;   % Find out what's worst acceptable (0 is acceptable)
opts.safeOpts.thresholdOffset = 1;
opts.safeOpts.searchCond = 1;
opts.safeOpts.onlyOptiDir = false;
opts.safeOpts.thresholdPer = 0.2;
opts.safeOpts.thresholdOrder = 1;
opts_lBO.maxIt = 100;
opts_lBO.sharedGP = 1;
opts_lBO.subspaceDim = 1;
opts_lBO.m = 5;

opts_lBO.oracle = 'random';
opts_lBO.alpha = 0.1;
opts_lBO.eta = 0.01;

fun = @(params) Ares_readWrite(params,pwd);
tic
[xopt,X,Y,DIM]=lineBO(hyp,inf_,mean_,cov_,lik_,acq, fun,cond,opts,opts_lBO,x0);
toc