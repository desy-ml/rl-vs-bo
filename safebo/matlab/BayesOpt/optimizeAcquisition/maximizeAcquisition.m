function [varargout]=maximizeAcquisition(hyp, inf_, mean_, cov_, lik_, x, y, acq_func,cond,opts,algo_data,varargin)
    % Find promising inputs depending on the options.
    % First posterior is calculated to avoid unecessary high computational
    % load due to matrix inversions

    D = size(cond,1);
    [~,~, post] = gp(hyp,inf_,mean_, cov_, lik_, x, y);
    if opts.moSaOpt
        [xnew, varargout{2},algo_data] = moSaOpt(hyp, inf_, mean_, cov_, lik_,x,y,post,acq_func,opts,algo_data);
        if nargout > 2
            varargout{3} = 0;
        end
    elseif opts.safeOpt
        [xnew, varargout{2},algo_data] = safeOpt(hyp, inf_, mean_, cov_, lik_,x,y,post,acq_func,opts,algo_data);
        if nargout > 2
            varargout{3} = 0;
        end
    else
        % find max acq value via grid search
        if D == 1
            x0 = varargin{1};
            nacq1=min(feval(acq_func{:},x0,hyp, inf_, mean_, cov_, lik_, x, post,y,opts.acqFunc,algo_data),0);
            nacq = nacq1;
            varargout{3}=nacq1;
            for i=1:1
                x0 = newInt(nacq,x0);
                nacq=feval(acq_func{:},x0,hyp, inf_, mean_, cov_, lik_, x, post, y,opts.acqFunc,algo_data);
            end
            nacq = min(nacq,0);
            [mnacq, arg] = min(nacq);
            xnew = x0(arg);
            varargout{2} = mnacq;
            
        else
            % find max acq value via DIRECT
            direct.showits = 0;
            direct.maxevals = 2000;
            direct.maxits = 500;
            direct.maxdeep = 500;
            %direct.tflag
            oldOpts.direct = direct;
            opts = getopts(oldOpts,opts);
            Problem.f=acq_func{:};
            [min_nacq, xnew] = Direct(Problem,cond,opts.direct,hyp, inf_, mean_, cov_, lik_, x, post,y,opts.acqFunc,algo_data);
            xnew = xnew';
            varargout{2} = min_nacq;
        end
    end
    varargout{nargout} = algo_data;
    if strcmp(algo_data.name, 'lineBO')
        x_vec = algo_data.x_vec;
        x_vec(algo_data.l) = xnew;
        varargout{1} = x_vec;
    else
        varargout{1} = xnew;
    end

function [x1]=newInt(nacq, x0)
    [~,arg] = min(nacq);
    if length(arg) > 1
        arg=arg(1);
    end
    if arg-1 < 1
        x1 = linspace(x0(1),x0(arg+1),1000)';
    elseif arg+1 > length(x0)
        x1 = linspace(x0(arg-1),x0(arg),1000)';
    else
        x1 = linspace(x0(arg-1),x0(arg+1),1000)';
    end

   