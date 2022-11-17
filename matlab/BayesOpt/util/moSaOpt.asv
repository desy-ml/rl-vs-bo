function [varargout] = moSaOpt(hyp, inf_, mean_, cov_, lik_, xt, yt,post,acq_func,opts,algo_data)
    % Find the parameters using MoSaOpt. It was used to optimize the LbSync system at Europen XFEL

    safeOpts = opts.safeOpts;
    acqOpts = opts.acqFunc;
    oldSafeOpts.thresholdOffset = 3;
    oldSafeOpts.thresholdPer = 0.2;
    oldSafeOpts.beta = 2;
    oldSafeOpts.onlyOptiDir = false;
    oldSafeOpts.searchCond = 1;
    
    safeOpts=getopts(oldSafeOpts,safeOpts);
    
    w = @(x) UCB(x,hyp, inf_, mean_, cov_, lik_, xt,post,yt,safeOpts,algo_data) + LCB(x,hyp, inf_, mean_, cov_, lik_, xt,post,yt,safeOpts,algo_data);
    xs_safe = safeOpts.xs_safe;
    threshold = safeOpts.threshold;
    thresh_per = safeOpts.thresholdPer;
    thresh_offset = safeOpts.thresholdOffset;
    searchCond = safeOpts.searchCond;

    if ~isfield(algo_data,'current_ymin') || algo_data.current_ymin ~= min(yt) || ~isfield(algo_data,'threshold_vec')
        algo_data.current_ymin = min(yt);
        thresh_max = max(0,threshold-min(yt)-thresh_offset);
        
        algo_data.threshold_vec = calc_threshold(thresh_max, thresh_per, xs_safe, safeOpts, algo_data);
    end

    if ~isfield(algo_data,'observed') || ~algo_data.observed
        U = -UCB(xs_safe, hyp, inf_, mean_, cov_, lik_, xt,post,yt,safeOpts, algo_data);
        I_S = U <= algo_data.threshold_vec;
        if isfield(algo_data,'S') && ~isempty(algo_data.S)
            S = union(xs_safe(I_S,:),algo_data.S,'rows');
        else
            S = xs_safe(I_S,:);
        end
    else
        S = algo_data.S;
    end
    algo_data.S = S;
    
    [min_u, I_minU] = min(yt);
    L = LCB(S,hyp, inf_, mean_, cov_, lik_, xt,post,yt,safeOpts,algo_data);
    I_M = L < min_u;
    M = S(I_M,:);


if length(algo_data.l) == 2
    bound_cond = boundary(xs_safe,1);
    [bound, A_ch] = boundary(S,1);
    G_re = xs_safe(bound_cond,:);
    G = setdiff(S(bound,:),G_re,'rows');
    if safeOpts.onlyOptiDir
        G = intersect(G,M,'rows');
    end
end

if length(algo_data.l) == 1
    G=S([1,end],:);
    dS = diff(S);
    d = xs_safe(2)-xs_safe(1);
    I = dS > d+eps;
    if any(I)
        I_t = find(I);
        I(I_t+1) = 1;
        G = union(G,S(I),'rows');
    end
    G_re = G;
end

M_re = setdiff(M,G_re,'rows');

if ~isempty(G)
    A = G; 
    nacq = w(A) / safeOpts.beta;
%     acq_val = w(A) / safeOpts.beta;
%     if all(abs(acq_val) < searchCond)
%         nacq = acq_val;
%     elseif length(algo_data.l) == 1
%         diff_p=xt(end,algo_data.l) - G;
%         c = abs(acq_val)>= searchCond;
%         [~,I]=min(abs(diff_p(c)));
%         A = G(c);
%         nacq = acq_val;
%         nacq(I) = min(acq_val)-1;
%     else
%         nacq = acq_val;
%     end
else 
    nacq = 0;
end


if all(abs(nacq) < searchCond) || algo_data.observed
    disp("EI used!!")
    A = S;
    nacq = feval(acq_func{:}, A, hyp, inf_, mean_, cov_, lik_, xt, post, yt, acqOpts, algo_data);
    algo_data.observed = true;
    algo_data.S = S;
else
    algo_data.observed = false;
end

if algo_data.plot 
    figure(4)
    if length(algo_data.l) == 1
        plot(S,zeros(size(S,1),1),'s','MarkerFaceColor','g')
        hold on
        plot(M_re,zeros(size(M_re,1),1),'s','MarkerFaceColor','r')
        plot(G_re,zeros(size(G_re,1),1),'s','MarkerFaceColor','b')
        xlim([xs_safe(1)-abs(xs_safe(end)-xs_safe(1))*0.1 xs_safe(end)+abs(xs_safe(end)-xs_safe(1))*0.1])
        hold off
        legend("$\mathcal{S}\setminus (\mathcal{G}\cup \mathcal{M})$","$\mathcal{M}$","$\mathcal{G}$",'interpreter','latex')
    else
        plot(S(:,1),S(:,2),'g.',G(:,1),G(:,2),'b*')
        hold on
        plot(xs_safe(:,1),xs_safe(:,2),':')
        plot(M(:,1),M(:,2),'r.')
        hold off
        legend("$\mathcal{S}\setminus (\mathcal{G}\cup \mathcal{M})$","$\mathcal{M}$","$\mathcal{G}$",'interpreter','latex')
    end
    %pause(2)
end

[m_nacq, I_x] = min(nacq);
varargout{2} = m_nacq ;
varargout{1} = A(I_x,:);
varargout{3} = algo_data;
end
    
    