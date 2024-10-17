clc;
clear;
close all;

U = 1.5;
D = 1;
L = 8;

%% Plot
force = readtable('postProcessing/forces/0/coefficient.dat');
figure; plot(force.x_Time,force.Cd/L); ylim([1.15 1.25]);
figure; plot(force.x_Time,force.Cl/L); ylim([-0.4 0.4]);

% ind = abs(force.Cl) > 1e-3;
ind = 4000:size(force,1);
figure; plot(force.Cd(ind),force.Cl(ind)); %axis([1.18 1.42 -0.3 0.3]);
figure; plot(force.Cl(ind),force.Cd(ind)); %axis([-0.35 0.35 1.3 1.5 ]);


%% FFT
startTime = 200;
time = force.x_Time;
t = time(time > startTime);
coeff = force.Cl(time > startTime);
coeffPrime = coeff - mean(coeff);
Fs = 1/(t(2)-t(1));

L = length(t);
Y = fft(coeffPrime);
P2 = abs(Y/L);
P1 = P2(1:floor(L/2)+1);
P1(2:end-1) = 2*P1(2:end-1);
f = Fs*(0:(L/2))/L;

figure; plot(f,P1,'.-');
xlim([0 1]);


%% Action time
[~,ind] = max(P1);
St = f(ind)*D/U

period = 1/f(ind)

nActions = 14
actionTime = period/nActions
actionTimeFraction = actionTime/period

%% Maximum jet flow rate
Qref = U*D; %flow rate encountering cylinder
Qjet_max = 0.1*Qref;
theta_jet = 10;
% l_jet = D/2*deg2rad(theta_jet)
l_jet = 2*D/2*sind(theta_jet/2)
U_jet_max = Qjet_max/l_jet


%% Mean of Cd
coeff = force.Cd(time > startTime);
mean(coeff)