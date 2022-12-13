function [xopt, X, Y, DIM] = lineBO(hyp,inf_,mean_,cov_,lik_,acq_func,obj_func,cond,opts_BO,opts_lineBO,varargin)
% Main that invokes bayesOptima with an subspaces instead of the whole
% parameter space
    
    D = size(cond,1);
    oldOpts_lineBO.maxIt = 50; % number of subspace iterations
    oldOpts_lineBO.sharedGP = true; % use a shared GPM
    oldOpts_lineBO.subspaceDim = 1; % dimensionality of the subspace (could be 1 or 2)
    oldOpts_lineBO.oracle = 'random'; % select oracle
    oldOpts_lineBO.dim_combinations = []; % restrict subspace combinations eg. [1;2;5;6] or [1,2;3,4;5,6]
    oldOpts_lineBO.m = []; % number of evaluations for "descent" oracle
    oldOpts_lineBO.alpha = 0.01; % stepsize also "descent" oracle
    oldOpts_lineBO.eta = 0.01;
    oldOpts_lineBO.obj_eval = @(y1,y2) y1 < y2-0.1; % could be a function handle with 2 inputs and boolean output: defines whether the subspace optimization improved the performance
    oldOpts_lineBO.maxProb = false; % maximization problem

    algoStruct.algo = 'lineBO';

    opts_lineBO = getopts(oldOpts_lineBO, opts_lineBO);
    
    if opts_lineBO.subspaceDim >= D
        error("Subspace dimension must be smaller than search space dimension")
    end
    if isempty(opts_lineBO.dim_combinations)
        opts_lineBO.dim_combinations=nchoosek(1:D,opts_lineBO.subspaceDim);
    end
    if isempty(opts_lineBO.m)
        opts_lineBO.m = 2 * size(opts_lineBO.dim_combinations,1);
    end

    if opts_BO.maxProb
        obj_eval = @max;
    else
        obj_eval = @min;
    end

    l = zeros(1,opts_lineBO.subspaceDim);
    l_t = buildObservedArray(opts_lineBO,D);
    c = 1;

    X = cell(opts_lineBO.maxIt,1);
    Y = cell(opts_lineBO.maxIt,1);

    DIM = zeros(opts_lineBO.maxIt,opts_lineBO.subspaceDim+2);
    DIM(:,opts_lineBO.subspaceDim+1) = false;
    
    if length(varargin) < 1 || isempty(varargin{1})
        randx = rand(1,D);
        xopt = bsxfun(@plus,bsxfun(@times,randx,(cond(:,2)'-cond(:,1)')),cond(:,1)');
    elseif length(varargin) == 1 
            xopt=varargin{1};
            if size(xopt,2) ~= length(cond)
                error("invalid x0")
            end
            xt = xopt;
            yt = obj_func(xt);
            yt_old = yt;
    elseif length(varargin) == 2
            xt = varargin{1};
            yt = varargin{2};
            [yt_old,temp] = obj_eval(yt);
            xopt = xt(temp,:);
    end
    

    for i=1:opts_lineBO.maxIt
        %fprintf("\n\n BO subspace no. %d/%d\n\n",i,opts_lineBO.maxIt)
%         [yopt,I] = obj_eval(yt);
%         xopt = xt(I,:);
        switch opts_lineBO.oracle
            case 'random'
                l = randomOrcale(l_t,opts_lineBO.dim_combinations);
            case 'coordinate'
                l = alignedOracle(i,D);
            case 'descent'
                [l,xt,yt] =  descentOracle(hyp,inf_,mean_,cov_,lik_,obj_func,xt,yt,xopt,opts_lineBO,cond,l_t);
        end

        algoStruct.subspace = l;
        
        if opts_lineBO.sharedGP
            algoStruct.post.x = xt;
            algoStruct.post.y = yt;
            
        else
            algoStruct.post.x = xopt;
            algoStruct.post.y = yopt;
        end
        algoStruct.post.yopt = yt_old;
        algoStruct.post.xopt = xopt;
        DIM(i,1:opts_lineBO.subspaceDim) = l;
        %fprintf("Dim: %d\n",algoStruct.subspace)
        %pause(1)
        [xopt,yt_new, xt, yt] = bayesOptima(hyp,inf_,mean_,cov_,lik_,acq_func,obj_func,cond,opts_BO,xopt,algoStruct);
%         yt_new = obj_eval(yt);
        %func = @(x) normcdf(x,yt_new,std_).*normpdf(x,yt_old,std_);
        if opts_lineBO.obj_eval(yt_new,yt_old)
            l_t = buildObservedArray(opts_lineBO,D);
            DIM(i,opts_lineBO.subspaceDim+1) = true;
            c = 1;
            yt_old = yt_new;
        end
        X{i} = xt;
        Y{i} = yt;
        DIM(i,end) = length(yt);
        l_t(c,:) = l;
        c = c + 1;
        if all(l_t ~= 0)
            buildObservedArray(opts_lineBO,D);
            disp("Optimum reached")
            fileID = fopen("/home/kaiserja/beegfs/ares-ea-v2/MSBO_optimum_reached",'w+');
            fprintf(fileID,' ');
            fclose(fileID);
            % break;
            exit;
        end
    end
end

function l = randomOrcale(l_t,combinations)
    l=combinations(randperm(size(combinations,1),1),:);
    while ~isempty(intersect(l, l_t,"rows")) || ~isempty(intersect(flip(l),l_t,"rows"))
        l=combinations(randperm(size(combinations,1),1),:);
    end
end

function l = alignedOracle(K,d)
    K = K-1;
    l = mod(K,d)+1;
end

function [l,xt,yt] = descentOracle(hyp,inf_,mean_,cov_,lik_,fun,xt,yt,xopt,opts,cond,l_t)
    combs = opts.dim_combinations;
    if size(combs,2) > 1
        error("descent oracle only for LineBO implemented")
    end
    combs = setdiff(combs,l_t,'rows');
    if size(combs,1) == 1
        l = combs;
        return
    end
    D_opt = size(combs,1);
    D = size(cond,1);
    intersec = intersect(opts.dim_combinations,l_t,'rows');
    m = opts.m - floor(opts.m/size(opts.dim_combinations,1)) * size(intersec,1);
    alpha = opts.alpha*ones(size(xopt,2),1);
    eta = opts.eta*ones(size(xopt,2),1);
    yopt = min(yt);
%     eta = (cond(:,2)-cond(:,1))/100;
%     alpha = (cond(:,2)-cond(:,1)) * alpha;
    xu = zeros(m,D);
    for i = 1:m    
        xs1 = xopt;
        t = floor((i-1)/D_opt);
        k = combs(i-t*D_opt);
        xs1(k) = xopt(k) + eta(k);
        xs = [xopt;xs1];
        if xs1(k) > cond(k,2)
            xs1(k) = xopt(k) - eta(k);
            xs = [xs1;xopt];
        end
            
        disp(xs);
        [~,~,mu,~,~,post] = gp(hyp,inf_,mean_,cov_,lik_,xt,yt,xs);
        v = post.L'\(repmat(post.sW,1,size(xs,1)).*feval(cov_{:},hyp.cov,xt,xs));
        K = feval(cov_{:},hyp.cov,xs)-v'*v;
        f=mvnrnd(mu,K);
        df = diff(f);
        new_val = xopt(k) - alpha(k) * df;

        if new_val < cond(k,1)
            new_val = cond(k,1);
        end
        if new_val > cond(k,2)
            new_val = cond(k,2);
        end
        xt1 = xopt;
        xt1(k) = new_val;
        yt1 = fun(xt1);
        xu(i,:) = xt1;
        if isempty(intersect(xt1,xt,"rows"))
            xt(end+1,:) = xt1;
            yt(end+1,:) = yt1;
%         else
%             t_str=load('/home/luebsen/master/master_thesis/matlab/own_lab/data/time.mat');
%             time = t_str.time;
%             time = time + 1;
%             save('/home/luebsen/master/master_thesis/matlab/own_lab/data/time.mat','time');
        end
        
    end
    [~,~,mu,~,~] = gp(hyp,inf_,mean_,cov_,lik_,xt,yt,xu);
    mu_g = (mu-yopt)./vecnorm(xopt-xu,2,2);
    [mu_gs,I] = sort(mu_g);
    disp([mu_gs I])
    disp(l_t)
    l = I(1);
    t = floor((l-1)/D_opt);
    l = l-t*D_opt;
    l = combs(l);
    while ~isempty(intersect(l,l_t,"rows")) 
        I(1) = [];
        mu_gs(1) = [];
        l = I(1);
        t = floor((l-1)/D_opt);
        l = l-t*D_opt;
        l = combs(l);
    end

end
    
    
function l_t=buildObservedArray(opts_lineBO,D)
    if isempty(opts_lineBO.dim_combinations)
        l_t = zeros(nchoosek(D,opts_lineBO.subspaceDim),opts_lineBO.subspaceDim);
    else
        l_t = zeros(size(opts_lineBO.dim_combinations,1),opts_lineBO.subspaceDim);
    end
end
