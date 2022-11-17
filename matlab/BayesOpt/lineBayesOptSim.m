%% Example script to optimize the simulated plant
% yalmip('clear'); 
clear all; close all;
%% Configuration
N = 2;  % Number of repetitions of G in system chain
N_L = 1;
%% Build model from parameters
lbsync = load('lbsync.mat');  % Load parameter database
tunit = 'seconds'; tunitexp = 0;  % Overwrite time scale (to evaluate effect on optimization problem)
scaling = 15;  % Exponent of the model output scaling, 0 = s, 12 = ps, 15 = fs, etc.
sys = build_laser_model(lbsync.sim.laser.origami, scaling, tunit);

% Plant
ctrl_gain = 1;
G = balreal(ss(series(sys.G_pzt, sys.G_l) / ctrl_gain));
G.u = 'u';
G.y = 'phi';
% Reference noise coloring filter
Fr = sys.Fr;
%Fr.P{1}(1) = -1e1 * 2*pi * 10^(-tunitexp);  % Change integral behaviour to frequency region of interest
Fr = balreal(ss(Fr));  % Alt: ss, balreal, prescale
Fr.D = zeros(size(Fr.D));  % Make proper

% Plant output disturbance coloring filter
Fd = sys.Fd;
%Fd.P{1}(1) = -1e1 * 2*pi * 10^(-tunitexp);  % Change integral behaviour to frequency region of interest
Fd = balreal(ss(Fd));  % Alt: ss, balreal, prescale
Fd.D = zeros(size(Fd.D));  % Make proper

Glaser = connect(G,Fd,sumblk('y = phi + d'),{'w','u'},{'y'});
Glaser2 = sys.G;

% Link model
if N_L > 0
    
    sys_link = lbsync.sim.link.short;
    sys_link.Fd.P{1}(end) = -1e-1 * 2*pi;
    G_pz = zpk(sys_link.G_pz);
    G_pz.P{1} = G_pz.P{1}(1:2);
    sys_link.G_pz = ss(G_pz);
    clear G_pz;
    
    sys_link = build_link_model(sys_link, scaling, tunit);
    Glink = sys_link.Gpade;
    %Glink = repmat({Glink}, 1, N_L);
end

% Connectivity
Fr.u = 'w(1)';
Fr.y = 'r';

if N_L < 1
    G = repmat({Glaser}, 1, N);
    sums = cell(1, N);
   
    for i = 1:N
        G{i}.u = {sprintf('w(%d)', i+1);sprintf('u(%d)', i)};
        G{i}.y = sprintf('y(%d)', i);
        if i == 1
            sums{i} = sumblk('e(1) = r - y(1)');
        else
            sums{i} = sumblk(sprintf('e(%1$d) = y(%2$d) - y(%1$d)', i, i-1));
        end
    end
    sums{end+1} = sumblk(sprintf('z = r - y(%d)', N));
else
    G = cell(1, N + N_L);
    G{1} = Glaser;
    for i=2:2:N+N_L
        G{i} = Glink;
        G{i+1} = Glaser;
    end
    for i = 1:N+N_L
        if mod(i,2) ~= 0
            G{i}.u = {sprintf('w(%d)', i+1) ;sprintf('u(%d)', i)};
            G{i}.y = sprintf('y(%d)', i);
            if i == 1
                sums{i} = sumblk('e(1) = r - y(1)');
            else
                sums{i} = sumblk(sprintf('e(%1$d) = y(%2$d) - y(%1$d)', i, i-1));
            end
        else
            G{i}.u = {sprintf('y(%d)',i-1); sprintf('w(%d)', i+1); sprintf('u(%d)', i)};
            G{i}.y = {sprintf('l(%d)', i/2);sprintf('y(%d)', i)};
            sums{i} = sumblk(sprintf('e(%d) = y(%d) - l(%d)', i, i-1, i/2));
        end
    end
    sums{end+1} = sumblk(sprintf('z = r - y(%d)', N+N_L));
end
Gg = connect(G{:}, Fr, sums{:}, {'u','w'},{'e','z'});

%%

Kp_max = 3e1;
Kp_min = 0.2;
Ki_max = 6e1;
Ki_min = 0;

cond_r=[Kp_min, Kp_max;
      Ki_min, Ki_max;
      %1.2, 3.8;
      0, 0.000105*350;
      0, 3;
      Kp_min, Kp_max;
      Ki_min, Ki_max;
      %1.2, 3.8;
      0, 0.000105*350;
      0, 3;
      Kp_min, Kp_max;
      Ki_min, Ki_max];
cond_r=cond_r(1:6,:);

cond = repmat([-1,1],size(cond_r,1),1);

% cond=[Kp_min, Kp_max;
%       Ki_min, Ki_max;
%       Kp_min, Kp_max;
%       Ki_min, Ki_max];

% cond=[Kp_min, Kp_max;
%       Ki_min, Ki_max;
%       Kp_min, Kp_max;
%       Ki_min, Ki_max;
%       Kp_min, Kp_max;
%       Ki_min, Ki_max];

% cond=[Kp_min, Kp_max;
%       Ki_min, Ki_max
%       ];

if N_L == 0
    scale = [1/5; 1/2];
%scale = [1/5;1/5];
    scale = repmat(scale,[N+N_L,1]);
else
    scale =[];
    scale1 = [1/5; 1/2];
    scale2 = [1/5; 1/5];
    for i = 1:N+N_L
        if mod(i,2)
            scale=[scale;scale1];
        else
            scale=[scale;scale2];
        end
    end
end

cov_matern = @(varargin)covMaternard(3,varargin);

inf_ = {@infGaussLik};
mean_ = {@meanConst};
lik_ = {@likGauss};
cov_ = {@(varargin)covMaternard(3,varargin{:})};
hyp.lik = log(0.1);
hyp.mean = 50;
hyp.cov = log([(cond(:,2)-cond(:,1)).*scale;15]);
acq = {@EI};

%x0 = [25,24,18,0.43,7,7.4];
%x0 = [3.80981, 24.67182, 0.80326, 12.43287, 21.95565, 23.44122];
%x0 = [23.71978, 13.51946, 1.79723, 24.50445, 9.57467, 26.47217, 2.45624, 8.95156, 3.19737, 12.91408];
x0 = forwardTransform(cond_r,[23.71978, 13.51946, 0.01, 2.5, 9.57467, 26.47217]);%, 0.02, 3, 3.19737, 12.91408]);
opts.plot=1;
opts.minFunc.mode=2;
opts.acqFunc.xi = 0.0;
opts.acqFunc.beta = 2;

opts.trainGP.acqVal = 2;%0.055;%0.5 %1D       %%% 0.05 D=1 with EI; 0.5 D = 1
opts.maxProb = 0;
opts.termCondAcq = 0.5;%0.05;%0.25;%0.5 %1D    %%% 0.05 D=1 with EI; 0.25 or 0.5 D=1; 0.2 D=2 with EI sf = 5
opts.maxIt = 500;
opts.trainGP.It = 10000;
opts.trainGP.train = 0;
opts.safeOpt = 1;
opts.moSaOpt = 1;
opts.safeOpts.threshold = 50;
opts.safeOpts.thresholdOffset = 100;
opts.safeOpts.searchCond = 3;
opts.safeOpts.onlyOptiDir = false;
opts.safeOpts.thresholdPer = 0.2;
opts.safeOpts.thresholdOrder = 1;
opts_lBO.maxIt = 100;
opts_lBO.sharedGP = 1;
opts_lBO.subspaceDim = 1;
%opts_lBO.obj_eval = @(y1,y2) y1 > y2+0.1;
%opts_lBO.dim_combinations = 5;
%opts.dir_timeData="/home/jannis/thesis_data/master_thesis/matlab/Temp";
%opts_lBO.dim_combinations = [2;6];
opts_lBO.oracle = 'descent';
opts_lBO.alpha = 0.2;
opts_lBO.eta = 0.1;

time = 0;
%save(opts.dir_timeData+"/time_data","time");
fun = @(params) connect_PI(params, Gg, [1/sys.k_phi 1/sys_link.k_phi],cond_r);
tic
[xopt,X,Y,DIM]=lineBO(hyp,inf_,mean_,cov_,lik_,acq, fun,cond,opts,opts_lBO,x0);
toc
%% 


function [y] = connect_PI(pi_params, Gg, scale,cond)
    pi_params = backTransform(cond,pi_params);
    N = length(pi_params)/2;
    C = cell(1,N);
    len_scale = length(scale);
    for i=1:N
        C{i} = pid(pi_params(1,2*i-1)*scale(len_scale-mod(i,len_scale)),pi_params(1,2*i)*scale(len_scale-mod(i,len_scale)));
        C{i}.y = sprintf('u(%d)', i);
        C{i}.u = sprintf('e(%d)', i);
    end
    Gcl = connect(Gg,C{:}, 'w','z');
    y = norm(Gcl,2)+randn(1)*0.5;
    if y == inf
        y = 1e4;
    end
end

function [x_transf]=forwardTransform(cond,x)
    mu_ = (cond(:,2)'+cond(:,1)')/2;
    slope_ = (cond(:,2)'-cond(:,1)')/2;
    x_transf = (x-mu_)./slope_;
end

function [x] = backTransform(cond,x_transf)
    mu_ = (cond(:,2)'+cond(:,1)')/2;
    slope_ = (cond(:,2)'-cond(:,1)')/2;
    x = (slope_.*x_transf)+mu_;
end


        
        
    
