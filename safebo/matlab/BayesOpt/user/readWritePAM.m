function data = readWritePAM(params,addr,duration,freq, lock_addr, jitter_addr,stepsize,t,cond)

% function to read BAM jitter and write new inputs into the Lbsync system
% the function is constructed to read the generated .npy files from
% Maximilian Schüttes home directory 

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
    disp("new backtransformed parameters: "+num2str(params));
    D = length(params);
    params_old = zeros(1,D);
    for i = 1:D
        data_str = doocsread(addr{i});
        params_old(i)=round(data_str.data,6);
    end
    counter=1:D;
    counter=counter(abs(params_old-params) > 10e-6);
    for i=counter
       writeData(addr{i},params(i),params_old(i),stepsize(i),t)
    end
    pause(duration+0.1)
    data = readData(addr,duration,freq,lock_addr,jitter_addr,params_old,params,stepsize,t);
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

function [data]=readData(addr,duration,freq,lock_addr,jitter_addr,params_old,params,stepsize,t)
    times = 1/freq;
    time = duration;
    iter = ceil(time/times);
    jitters = zeros(iter,length(jitter_addr));
    for i = 1:iter
            if ~checkLockStatus(lock_addr)
                dims_=1:size(params_old,2);
                dims_=dims_(abs(params_old-params) > 10e-6);
                for k=dims_
                   writeData(addr{i},params_old(i),params(i),stepsize(i),t)
                end
                warning("Some system(s) are not locked")
                count_str=load('/home/luebsen/master/master_thesis/matlab/own_lab/data/counter_unlock.mat');
                counter = count_str.counter;
                counter = counter + 1;
                save('/home/luebsen/master/master_thesis/matlab/own_lab/data/counter_unlock.mat','counter');
                jit_str = load('/home/luebsen/master/master_thesis/matlab/own_lab/data/jitter_data','jitter_data');
                jitter_data = jit_str.jitter_data;
                jitter_data(end+1,:) = zeros(1,size(jitter_addr,1)+1);
                save('/home/luebsen/master/master_thesis/matlab/own_lab/data/jitter_data.mat','jitter_data');
                data = input("set value manually\n");
                return
            end

           for l = 1:length(jitter_addr)
               temp_str = doocsread(jitter_addr{l});
               jitters(i,l) = temp_str.data;
           end
           pause(times)
    end
    filename = getFilename("/home/mschuet/spbonline/20220509_pam_data");
    pam_measure = readPAM(filename); %%%% important %%%%
    jitter = std(pam_measure,0,2);
    data = mean(jitter);
    std_dev = std(jitter);
    jit_str = load('/home/luebsen/master/master_thesis/matlab/own_lab/data/jitter_data','jitter_data');
    jitter_data = jit_str.jitter_data;
    jitter_data(end+1,:) = [mean(jitters,1),std_dev];
    save('/home/luebsen/master/master_thesis/matlab/own_lab/data/jitter_data.mat','jitter_data');
    disp("measured value: " + num2str(data))
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

function str = getFilename(in_dir)
    %in_dir = "/home/mschuet/spbonline/20220509_pam_data";
    fileInfo = dir(in_dir);
    files = {fileInfo.name};
    files = files(3:end);
    id=strfind(files,'_');
    arr = zeros(length(id),1);
    for i = 1:length(id)
        temp_id = id{i};
        temp = files{i};
        arr(i) = str2double(temp(temp_id(end)+1:end-4));
    end
    [~,I] = sort(arr);
    str = in_dir+"/"+files{I(end)};
end
    