function X = gradient_descent(X,f,opts,varargin)
    oldOpts.showIts = 0;
    if isempty(opts) || ~isstruct(opts)
        opts = oldOpts;
    else
        opts=getopts(oldOpts,opts);
    end

    if length(varargin) > 7
        error("To many inputs")
    elseif length(varargin) < 7
        lp.cov = 0.01;
        lp.lik = 0.01;
        len = length(varargin);
    else
        lp = varargin{end};
        len = length(varargin)-1;
    end
    i = 0;
    disp("Iterations        Gradients           lp")
    [nlZ, dnlZ] = feval(f,X, varargin{1:len});
    while ((any(abs(dnlZ.cov) > 0.01) || any(abs(dnlZ.lik) > 0.01))) && i < 2000

        X.cov = X.cov - min(lp.cov,0.1/(min(abs(dnlZ.cov)))) * dnlZ.cov;
        X.lik = X.lik - min(lp.lik,0.1/(min(abs(dnlZ.lik)))) * dnlZ.lik;

        i = i + 1;
        if opts.showIts
            disp(num2str(i)+"       "+num2str(dnlZ.cov')+" "+num2str(dnlZ.lik')+ "    "+num2str(lp.cov)+" "+num2str(lp.lik))
        end
        dnlZ_old = dnlZ;
        [nlZ, dnlZ] = feval(f,X, varargin{1:len});
        if isstruct(lp)
            lp.cov=change_lp(lp.cov,dnlZ.cov,dnlZ_old.cov);
            lp.lik=change_lp(lp.lik,dnlZ.lik,dnlZ_old.lik);
        else
            lp=change_lp(lp,dnlZ.cov,dnlZ_old.cov);
        end
    end
end

function lp = change_lp(lp,dnlZ_old, dnlZ)
    if any((dnlZ_old<0)~=(dnlZ<0))
        lp = lp/2;
    else
        return
    end
end

function opts=getopts(opts,newOpts)
    ch1=fieldnames(newOpts);
    ch2=fieldnames(opts);
    for i=1:length(ch1)
        for j=1:length(ch2)
            if strcmp(ch1{i},ch2{j})
                if isstruct(opts.(ch1{i})) && isstruct(newOpts.(ch2{j}))
                    opts.(ch1{i})=getopts(opts.(ch1{i}),opts.(ch2{j}));
                else
                    opts.(ch1{i})=newOpts.(ch2{j});
                end 
            end
        end
    end
end
