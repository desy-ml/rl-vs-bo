function [threshold_vec] = calc_threshold(tresh_max, tresh_per, xs, opts, algo_data)

    oldOpts.thresholdOrder = 1;
    opts = getopts(oldOpts,opts);

    threshold = opts.threshold;
    n = opts.thresholdOrder;
    %threshold = algo_data.current_ymin + 20;
    if length(algo_data.l) == 1
        threshold_vec = ones(size(xs,1),1)*threshold;
        l = floor(length(xs)*tresh_per);
        slope = -tresh_max+(tresh_max)/(xs(l)-xs(1))^n*(xs(1:l)-xs(1)).^n;
        threshold_vec(1:l) = threshold_vec(1:l)+slope;
        threshold_vec(end-l+1:end)=threshold_vec(end-l+1:end)+flip(slope,1);
        if algo_data.plot
            figure(3)
            plot(xs,threshold_vec)
            ylabel("threshold","Interpreter","latex")
            xlabel("position","Interpreter","latex")
        end
    else
        threshold_vec = ones(size(xs,1),1)*threshold;
        lx = floor(size(xs,1)*tresh_per);
        ly = floor(sqrt(size(xs,1))*tresh_per);
        imax = sqrt(size(xs,1));
        for i=1:imax/2
            v = (i-1)*imax;
            if v < lx
                x = (xs(v+1:v+imax,1)-xs(1,1));
            else
                x = (xs(lx,1)-xs(1,1));
            end
            y = ones(imax,1)*(xs(v+ly,2)-xs(v+1,2)); 
            y(1:ly) = (xs(v+1:v+ly,2)-xs(v+1,2));
            y(end-ly+1:end) = flip(y(1:ly),1);
            denum = (xs(lx,1)-xs(1,1))*(xs(v+ly,2)-xs(v+1,2));
            slope = tresh_max/(denum)^n*(x.*y).^n;

            threshold_vec(v+1:v+imax,1)=threshold_vec(v+1:v+imax,1)-tresh_max+slope;
            threshold_vec(end-v-imax+1:end-v)=threshold_vec(end-v-imax+1:end-v)-tresh_max+flip(slope,1);
        end
        if algo_data.plot
            figure(3)
            plot3(xs(:,1),xs(:,2),threshold_vec(:,1))
            zlabel("threshold","Interpreter","latex")
            xlabel("subdimension 1",'Interpreter','latex')
            ylabel("subdimension 2",'Interpreter','latex')
        end
    end
end