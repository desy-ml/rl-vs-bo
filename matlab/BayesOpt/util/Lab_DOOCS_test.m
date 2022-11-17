MENHIR = {'LAB.SYNC/LASER.LOCK/26A1.L3.MENHIR/ADV_CTRL_MANAGER.0.PID.1.P_PARAM.WR',
    'LAB.SYNC/LASER.LOCK/26A1.L3.MENHIR/ADV_CTRL_MANAGER.0.PID.1.I_PARAM.WR',
    'LAB.SYNC/LASER.LOCK/26A1.L3.MENHIR/RF_HARM_PHASE.CONV.A1.WR'
    };

Link = {'LAB.SYNC/LINK.LOCK/26A.AMC7.CONTROLLER/LSU.1.SS_PID.P_PARAM.WR',...
    'LAB.SYNC/LINK.LOCK/26A.AMC7.CONTROLLER/LSU.1.SS_PID.I_PARAM.WR',...
    'LAB.SYNC/LINK.LOCK/26A.AMC7.CONTROLLER/LSU.1.CALIB.OXC_DET.CC_FS_TO_V.RD'};

ORIGAMI = {'LAB.SYNC/LASER.LOCK/26A2.L2.ORIGAMI10/ADV_CTRL_MANAGER.0.PID.2.P_PARAM.WR',...
    'LAB.SYNC/LASER.LOCK/26A2.L2.ORIGAMI10/ADV_CTRL_MANAGER.0.PID.2.I_PARAM.WR',...
    'LAB.SYNC/LASER.LOCK/26A2.L2.ORIGAMI10/RF_HARM_PHASE.CONV.A1.WR'};

% Z = {'LAB.SYNC/LINK.LOCK/26A.AMC7.CONTROLLER/LSU.0.CTRL_IN.STD_DEV.RD'}; %kphi = 2565;
% Z = {'LAB.SYNC/LASER.LOCK/26A1.L3.MENHIR/DC8.STD_DEV.RD'}; %kphi = 9000;
% Z = {'XFEL.SDIAG/BAM/1932S.TL/EXPERT_STATISTICS.MACRO_PULSE.interMpStdev.1'};

addresses={'LAB.SYNC/LINK.LOCK/26A.AMC7.CONTROLLER/LSU.1.SS_PID.P_PARAM.WR',...
    'LAB.SYNC/LINK.LOCK/26A.AMC7.CONTROLLER/LSU.1.SS_PID.I_PARAM.WR',...
    'LAB.SYNC/LASER.LOCK/26A1.L1.ORIGAMI15/ADV_CTRL_MANAGER.0.PID.2.P_PARAM.WR',...
    'LAB.SYNC/LASER.LOCK/26A1.L1.ORIGAMI15/ADV_CTRL_MANAGER.0.PID.2.I_PARAM.WR',...
    'LAB.SYNC/LASER.LOCK/26A1.L1.ORIGAMI15/DCS_7.SPEC',...
    'LAB.SYNC/LASER.LOCK/26A1.L1.ORIGAMI15/DCS_8.SPEC'
    };


lock_status={'LAB.SYNC/LASER.LOCK/26A1.L1.ORIGAMI15/LOCK_STATUS.VALUE.RD',...
    'LAB.SYNC/LASER.LOCK/26A1.L3.MENHIR/LOCK_STATUS.VALUE.RD',...
    'LAB.SYNC/LINK.LOCK/26A.AMC7.CONTROLLER/LSU.1.LOCK_STATUS.VALUE.RD'};

jitter_addr={'LAB.SYNC/LASER.LOCK/26A1.L3.MENHIR/CURRENT_INPUT_JITTER.RD',...
    'LAB.SYNC/LINK.LOCK/26A.AMC7.CONTROLLER/LSU.1.TIMING_JITTER_FS.RD',...
    'LAB.SYNC/LASER.LOCK/26A1.L1.ORIGAMI15/CURRENT_INPUT_JITTER.RD'};


c = addresses(end-2:end);
for i = 1:length(c)
    data_struct = doocsread(c{i});
    data_struct.data
end
    

address_J = addresses(end-1:end);
times = 0.10;
time = 60;
iter = ceil(time/times);
jitter = zeros(iter,1);
%mes=doocswrite(Link{2},0.0007);
for i = 1:iter
   signals = zeros(32768,2);
   timest = [0,1];
   while timest(1) ~= timest(2)
       for j = 1:length(address_J)
            data_str = doocsread(address_J{mod(j-1,2)+1});
            signals(:,j) = data_str.data.d_spect_array_val;
            timest(j)=data_str.timestamp;
    %         pause(1)
       end
   end
   jitter(i) = 4000*std(diff(signals,1,2),1); 
   pause(times)
end

fig1=figure(1);
t = linspace(0,time,iter);
plot(t,jitter)
ylabel("J [fs]",'interpreter','latex')
xlabel("t [s]", 'interpreter','latex')
grid on

% figure(2)
% ts = 5;
% f=zeros(iter-ts,1);
% for i=ts+1:iter
%     f(i-ts)=mean(jitter(ts:i));
% end
% mu = mean(jitter(ts:end))*ones(iter-ts,1);
% plot(t(ts+1:end),f,t(ts+1:end),mu)
% hold on
% plot(t(ts+1:end),mu+std(jitter(ts:end)),'k--',t(ts+1:end),mu-std(jitter(ts:end)),'k--')
% hold off
% ylabel("J [fs]",'interpreter','latex')
% xlabel("t [s]", 'interpreter','latex')
% grid on
% legend("mean(J(t))","$\mu$","$\mu \pm \sigma$",'interpreter','latex')

figure(2)
ts = 1/times;
f=zeros(floor(iter/ts),1);
for i=1:floor(iter/ts)
    f(i)=mean(jitter((i-1)*ts+1:i*ts));
end
disp(std(f))
mu = mean(jitter(ts:end))*ones(iter-ts,1);
plot(t(ts:ts:end),f,t(ts+1:end),mu)
hold on
plot(t(ts+1:end),mu+std(f),'k--',t(ts+1:end),mu-std(f),'k--')
hold off
ylabel("J [fs]",'interpreter','latex')
xlabel("t [s]", 'interpreter','latex')
grid on
legend("mean(J(t))","$\mu$","$\mu \pm \sigma$",'interpreter','latex')

figure(3)
ts = 1/times;
f=zeros(floor(iter/ts),1);
for i=1:floor(iter/ts)
    f(i)=median(jitter((i-1)*ts+1:i*ts));
end
disp(std(f))
mu = median(jitter(ts:end))*ones(iter-ts,1);
plot(t(ts:ts:end),f,t(ts+1:end),mu)
hold on
plot(t(ts+1:end),mu+std(f),'k--',t(ts+1:end),mu-std(f),'k--')
hold off
ylabel("J [fs]",'interpreter','latex')
xlabel("t [s]", 'interpreter','latex')
grid on
legend("median(J(t))","median",'interpreter','latex')

% jitter2.timestamp