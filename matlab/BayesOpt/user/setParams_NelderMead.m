function varargout= setParams_NelderMead(params,addr,duration,freq, lock_addr, jitter_addr,stepsize,t,cond,kphi)
    
    % function that read the jitter from the lab test plant and writes new
    % parameters to the system. 
    % It is adapted to the Nelder-Mead algorithm (fminsearch)
    % params: column array of input parameters
    % addr: doocs addresses of PI parameters (same order thab PI values in
    % params array
    % duration: measurement time
    % freq: frequency to pull new data
    % lock_addr: addresses of the lock status of each subsystem
    % jitter_addr: addresses of in-loop jitters
    % stepsize and t: defines how fast the params are changed 'stepsize' denotes
    % the largest jump within time 't'
    % kphi: constant to determine the jitter in [fs]
    % dir: directory where data like in-loop jitters should be saved
    % mean_: defines whether the mean (mean_=1) or median (mean_=0) should be
    % applied over the jitters
    D = length(params);
    params_old = zeros(1,D);
    for i = 1:D
        data_str = doocsread(addr{i});
        params_old(i)=round(data_str.data,6);
    end
%     params = transformParams(params,cond);
    disp("parameters"+string(params)) 
    counter=1:D;
    counter=counter(abs(params_old-params) > 10e-6);
    %disp(counter)
    for i=counter
       writeData(addr{i},params(i),params_old(i),stepsize,t)
    end
    pause(0.1)
    %pause(2)
    [data,jitters] = readData(addr(end-1:end),duration,freq,lock_addr,jitter_addr,kphi);
    varargout{1} = data;
    if nargout == 2
       varargout{2} = jitters;
    end
end

function writeData(addr,param,param_old,stepsize,t)
    num_step=fix((param-param_old)/stepsize);
    for i = 1:abs(num_step)
        if num_step > 0
            m=doocswrite(addr,param_old+i*stepsize);
        else
            m=doocswrite(addr,param_old-i*stepsize);
        end
        pause(t)
    end
    m=doocswrite(addr,param);
end

function [data,jitters]=readData(addr,duration,freq,lock_addr,jitter_addr,kphi)
    times = 1/freq;
    time = duration;
    iter = ceil(time/times);
    jitter = zeros(iter,1);
    jitters = zeros(iter,length(jitter_addr));
    for i = 1:iter
         if ~checkLockStatus(lock_addr)
             count_str=load('/home/luebsen/master/master_thesis/matlab/own_lab/test/counter_unlock.mat');
             counter = count_str.counter;
             counter = counter + 1;
             save('/home/luebsen/master/master_thesis/matlab/own_lab/test/counter_unlock.mat','counter');
             st=tic;
            while ~checkLockStatus(lock_addr)
                pause(3)
                if toc(st) > 31
                    s = input("Can't lock system, set value manually by writing 'user' or continue by pressing 'enter': ",'s');
                    if strcmp(s,'user')
                        data = input("Set value: ");
                        jitters=100*ones(1,length(jitter_addr)+1);
                    else
                        st = tic;
                        continue;
                    end
                    return
                end
            end
         end

        timest=[0,1];
        signals = zeros([32768,2]);
        while timest(1) ~= timest(2)
           for j = 1:length(addr)
                data_str = doocsread(addr{j});
                signals(:,j) = data_str.data.d_spect_array_val;
                timest(j)=data_str.timestamp;
           end
    %         pause(1)
        end
        jitter(i) = kphi*std(diff(signals,1,2),1); 
        for l = 1:length(jitter_addr)
           temp_str = doocsread(jitter_addr{l});
           jitters(i,l) = temp_str.data;
        end
        pause(times)
   end
   data(j) = mean(jitter);
   std_dev(j) = std(jitter);
    std_dev = std_dev(j);
    data = data(j);
    val_str=load('/home/luebsen/bayesopt/bayesianoptimization/data/val.mat');
    val = val_str.val;
    val(end+1,1) = data;
    val(end,2) = std_dev;
    save('/home/luebsen/bayesopt/bayesianoptimization/data/val.mat','val');
    jitters = [mean(jitters,1),std_dev];
    disp("measured value: "+string(data))
end

function b=checkLockStatus(lock_addr)
    c = 0;
    for i =1:length(lock_addr)
        c_str = doocsread(lock_addr{i});
        c = c + c_str.data;
    end
    if c == 0
        b = true;
    else
        b = false;
    end
end  

function x=transformParams(y,cond)
    x=(cond(:,2)'-cond(:,1)').*(sin(y)+1)/2 + cond(:,1)';
end
