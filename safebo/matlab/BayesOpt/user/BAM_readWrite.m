function data = BAM_readWrite(params,addr,duration,freq, lock_addr, jitter_addr,stepsize,t,kphi,cond,dir,mean_)

% function to read BAM jitter and write new inputs into the Lbsync system
% the function is constructed to read the inter marco pulse standard
% deviation

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
% mean_: defines if the mean (mean_=1) or median (mean_=0) should be
% applied over the jitters

    params = backwardCoordTransf(cond,params);
    D = length(params);
    params_old = zeros(1,D);
    for i = 1:D
        data_str = doocsread(addr{i});
        params_old(i)=round(data_str.data,6);
    end
    counter=1:D;
    counter=counter(abs(params_old-params) > 10e-6);
    disp(params_old)
    disp(params)

    for i=counter
       writeData(addr{i},params(i),params_old(i),stepsize(i),t)
    end
    pause(0.3)
    data = readData(addr,duration,freq,lock_addr,jitter_addr,params_old,params,kphi,dir,mean_,stepsize,t);
end

function writeData(addr,param,param_old,stepsize,t)
    num_step=fix((param-param_old)/stepsize);
    disp(num_step)
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

function [jitter]=readData(addr,duration,freq,lock_addr,jitter_addr,params_old,params,kphi,dir_,mean_,stepsize,t)
    times = 1/freq;
    time = duration;
    iter = ceil(time/times);
%     jitter = zeros(iter,1);
    jitters = zeros(iter,length(jitter_addr));
    for i = 1:iter
         if ~checkLockStatus(lock_addr)
                dims_=1:size(params_old,2);
                dims_=dims_(abs(params_old-params) > 10e-6);
                for k=dims_
                   writeData(addr{i},params_old(i),params(i),stepsize(i),t)
                end
                warning("Some system(s) is/are not locked")
                count_str=load(dir_+'/counter_unlock.mat');
                counter = count_str.counter;
                counter = counter + 1;
                save(dir_+'/counter_unlock.mat','counter');
                jit_str = load(dir_+'/jitter_data.mat','jitter_data');
                jitter_data = jit_str.jitter_data;
                jitter_data(end+1,:) = zeros(1,size(jitter_addr,1)+1);
                save(dir_+'/jitter_data.mat','jitter_data');
                data = input("set value manually\n");
                return
          end
    
       for l = 1:length(jitter_addr)
           temp_str = doocsread(jitter_addr{l});
           jitters(i,l) = temp_str.data;
       end
       pause(times)
    end
    data_str = doocsread(addr{end});
    jitter = data_str.data*kphi;
    num = length(dir("/home/luebsen/master/master_thesis/matlab/own_lab/BAM_jitters"));
    save("/home/luebsen/master/master_thesis/matlab/own_lab/BAM_jitters/"+sprintf("jitter_data_%d",num))
%     if mean_
%         data = mean(jitter);
%     else
%         data = median(jitter);
%     end
%     
%     std_dev = std(jitter);
    jit_str = load(dir_+'/jitter_data','jitter_data');
    jitter_data = jit_str.jitter_data;
    jitter_data(end+1,:) = [mean(jitters,1),jitter];
    save(dir_+'/jitter_data.mat','jitter_data');
    disp("measured value: " + num2str(jitter))
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
    b = 1;
end  

