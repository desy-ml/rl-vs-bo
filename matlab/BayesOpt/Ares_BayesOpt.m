

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
hyp.lik = log(0.1);
hyp.mean = 0;
hyp.cov = log([0.1;0.1;0.1;0.1;0.1;5]);
acq = {@EI};

%x0 = [25,24,18,0.43,7,7.4];
%x0 = [3.80981, 24.67182, 0.80326, 12.43287, 21.95565, 23.44122];
%x0 = [23.71978, 13.51946, 1.79723, 24.50445, 9.57467, 26.47217, 2.45624, 8.95156, 3.19737, 12.91408];
<<<<<<< HEAD
x0 = forwardCoordTransform(cond_r,[23.71978, 13.51946, 0.01, 2.5, 9.57467, 26.47217]);%, 0.02, 3, 3.19737, 12.91408]);
opts.plot=1;
=======
% Normalised version of [10, -10, 0, 10, 0]
x0 = [-0.7222, 0.7222, 0.0, -0.7222, 0.0] %forwardTransform(cond_r,[23.71978]);%, 0.02, 3, 3.19737, 12.91408]);
opts.plot=0;
>>>>>>> 910a38cda74e4f45db28e2b40ceae0fbf3a02482
opts.minFunc.mode=2;
opts.acqFunc.xi = 0.0;
opts.acqFunc.beta = 2;

opts.trainGP.acqVal = 2;%0.055;%0.5 %1D       %%% 0.05 D=1 with EI; 0.5 D = 1
opts.maxProb = 0;
opts.termCondAcq = 0.3;%0.05;%0.25;%0.5 %1D    %%% 0.05 D=1 with EI; 0.25 or 0.5 D=1; 0.2 D=2 with EI sf = 5
opts.maxIt = 500;
opts.trainGP.It = 10000;
opts.trainGP.train = 0;
opts.safeOpt = 1;
opts.moSaOpt = 1;
opts.safeOpts.threshold = 0;   % Find out what's worst acceptable (0 is acceptable)
opts.safeOpts.thresholdOffset = 100;
opts.safeOpts.searchCond = 3;
opts.safeOpts.onlyOptiDir = false;
opts.safeOpts.thresholdPer = 0.2;
opts.safeOpts.thresholdOrder = 1;
opts_lBO.maxIt = 100;
opts_lBO.sharedGP = 1;
opts_lBO.subspaceDim = 1;

opts_lBO.oracle = 'descent';
opts_lBO.alpha = 0.2;
opts_lBO.eta = 0.1;

fun = @(params) Ares_readWrite(params,pwd);
tic
[xopt,X,Y,DIM]=lineBO(hyp,inf_,mean_,cov_,lik_,acq, fun,cond,opts,opts_lBO,x0);
toc