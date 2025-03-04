%%=============================================================
%The file is used to generate the Figure 2 of the paper:
%
%Zhe Wang, Jiayi Zhang, Hao Lei, Dusit Niyato, and Bo Ai, "Optimal Bilinear Equalizer Beamforming Design for Cell-Free Massive MIMO Networks with Arbitrary Channel Estimators,"
%IEEE Transactions on Vehicular Technology, to appear, 2024, %doi: 10.1109/TVT.2024.3520500.
%
%Download article: https://arxiv.org/abs/2503.00763 or https://ieeexplore.ieee.org/document/10810748
%
%License: This code is licensed under the GPLv2 license. If you in any way
%use this code for research that results in publications, please cite our
%paper as described above.
%%=============================================================

clc
clear all
close all

tic
M = 40;
N = 4;
K = 10;


nbrOfRealizations = 1000;
nbrOfSetups = 500;
tau_p = 1;
tau_c = 200;


%Uplink transmit power per UE (W)
p = 0.2; %200 mW
%Create the power vector for all UEs (The uplink power is the same
%(p)at each UE)

SE_Distributed_OBE_MMSE_Monte_Rician = zeros(K,nbrOfSetups);
SE_Distributed_OBE_LS_Monte_Rician = zeros(K,nbrOfSetups);

SE_Distributed_OBE_MMSE_Monte_Rayleigh = zeros(K,nbrOfSetups);
SE_Distributed_OBE_LS_Monte_Rayleigh = zeros(K,nbrOfSetups);

pv = p*ones(1,K);

% Rician fading channel
probLoS_Rician = ones(M,1);
probLoS_Rayleigh = zeros(M,1);

A_CE_matrix_LS = zeros(N,N,M,K);


for m = 1:M
    for kk = 1:K

        A_CE_matrix_LS(:,:,m,kk) = 1/(sqrt(pv(kk))*tau_p)*eye(N);

    end

end


% % % % % % % ----Parallel computing
core_number = 4;           
parpool('local',core_number);
% % % % Starting parallel pool (parpool) using the 'local' profile ...



parfor i = 1:nbrOfSetups


    [R_AP_Rayleigh,R_AP_Rician,H_LoS_Single_real_Rician] = functionGenerateSetupDeploy_Rician_Rayleigh(M,K,N,1,1,probLoS_Rician,probLoS_Rayleigh);



    %--Rician
    [H_Rician,H_LoS_Rician] = functionChannelGeneration(R_AP_Rician,H_LoS_Single_real_Rician,M,K,N,nbrOfRealizations);


    A_singleLayer = reshape(repmat(eye(M),1,K),M,M,K);
    [Pset_Rician] = functionPilotAllocation(R_AP_Rician,H_LoS_Single_real_Rician,A_singleLayer,M,K,N,tau_p,pv);

    [AMMSE_Rician] = functionMMSEChannelEstimator(R_AP_Rician,pv,M,K,N,tau_p,Pset_Rician);
    


    [Phi_Rician,~] = functionMatrixGeneration(AMMSE_Rician,R_AP_Rician,pv,M,K,N,tau_p,Pset_Rician);


    [Hhat_MMSE_Rician] = functionChannelEstimates(AMMSE_Rician,H_LoS_Rician,H_Rician,nbrOfRealizations,M,K,N,tau_p,pv,Pset_Rician);
    [Hhat_LS_Rician] = functionChannelEstimates(A_CE_matrix_LS,H_LoS_Rician,H_Rician,nbrOfRealizations,M,K,N,tau_p,pv,Pset_Rician);


    %---Distributed
    [V_OBE_Combining_Distributed_Rician,W_OBE_matrix_Distributed_Rician] = functionOBE_Combining_Distributed(H_LoS_Rician,Hhat_MMSE_Rician,AMMSE_Rician,Phi_Rician,R_AP_Rician,Pset_Rician,M,N,K,pv,tau_p,nbrOfRealizations);
    [V_OBE_Combining_Distributed_LS_Rician,W_OBE_matrix_Distributed_LS_Rician] = functionOBE_Combining_Distributed(H_LoS_Rician,Hhat_LS_Rician,A_CE_matrix_LS,Phi_Rician,R_AP_Rician,Pset_Rician,M,N,K,pv,tau_p,nbrOfRealizations);


    [SE_OBE_Monte_LSFD_MMSE_Rician,SE_OBE_Monte_MMSE_Rician] = functionComputeSE_Distributed_Monte(H_Rician,V_OBE_Combining_Distributed_Rician,tau_c,tau_p,nbrOfRealizations,N,K,M,pv);
    [SE_OBE_Monte_LSFD_LS_Rician,SE_OBE_Monte_LS_Rician] = functionComputeSE_Distributed_Monte(H_Rician,V_OBE_Combining_Distributed_LS_Rician,tau_c,tau_p,nbrOfRealizations,N,K,M,pv);



    SE_Distributed_OBE_MMSE_Monte_Rician(:,i) = SE_OBE_Monte_MMSE_Rician;
    SE_Distributed_OBE_LS_Monte_Rician(:,i) = SE_OBE_Monte_LS_Rician;

    %--Rayleigh

    H_LoS_Single_real_Rayleigh = zeros(M*N,K);

    [H_Rayleigh,H_LoS_Rayleigh] = functionChannelGeneration(R_AP_Rayleigh,H_LoS_Single_real_Rayleigh,M,K,N,nbrOfRealizations);


    A_singleLayer = reshape(repmat(eye(M),1,K),M,M,K);

    [Pset_Rayleigh] = functionPilotAllocation(R_AP_Rayleigh,H_LoS_Single_real_Rayleigh,A_singleLayer,M,K,N,tau_p,pv);
    [AMMSE_Rayleigh] = functionMMSEChannelEstimator(R_AP_Rayleigh,pv,M,K,N,tau_p,Pset_Rayleigh);


    [Phi_Rayleigh,~] = functionMatrixGeneration(AMMSE_Rayleigh,R_AP_Rayleigh,pv,M,K,N,tau_p,Pset_Rayleigh);


    [Hhat_MMSE_Rayleigh] = functionChannelEstimates(AMMSE_Rayleigh,H_LoS_Rayleigh,H_Rayleigh,nbrOfRealizations,M,K,N,tau_p,pv,Pset_Rayleigh);
    [Hhat_LS_Rayleigh] = functionChannelEstimates(A_CE_matrix_LS,H_LoS_Rayleigh,H_Rayleigh,nbrOfRealizations,M,K,N,tau_p,pv,Pset_Rayleigh);


    %---Distributed
    [V_OBE_Combining_Distributed_Rayleigh,W_OBE_matrix_Distributed_Rayleigh] = functionOBE_Combining_Distributed(H_LoS_Rayleigh,Hhat_MMSE_Rayleigh,AMMSE_Rayleigh,Phi_Rayleigh,R_AP_Rayleigh,Pset_Rayleigh,M,N,K,pv,tau_p,nbrOfRealizations);
    [V_OBE_Combining_Distributed_LS_Rayleigh,W_OBE_matrix_Distributed_LS_Rayleigh] = functionOBE_Combining_Distributed(H_LoS_Rayleigh,Hhat_LS_Rayleigh,A_CE_matrix_LS,Phi_Rayleigh,R_AP_Rayleigh,Pset_Rayleigh,M,N,K,pv,tau_p,nbrOfRealizations);


    [SE_OBE_Monte_LSFD_MMSE_Rayleigh,SE_OBE_Monte_MMSE_Rayleigh] = functionComputeSE_Distributed_Monte(H_Rayleigh,V_OBE_Combining_Distributed_Rayleigh,tau_c,tau_p,nbrOfRealizations,N,K,M,pv);
    [SE_OBE_Monte_LSFD_LS_Rayleigh,SE_OBE_Monte_LS_Rayleigh] = functionComputeSE_Distributed_Monte(H_Rayleigh,V_OBE_Combining_Distributed_LS_Rayleigh,tau_c,tau_p,nbrOfRealizations,N,K,M,pv);



    SE_Distributed_OBE_MMSE_Monte_Rayleigh(:,i) = SE_OBE_Monte_MMSE_Rayleigh;
    SE_Distributed_OBE_LS_Monte_Rayleigh(:,i) = SE_OBE_Monte_LS_Rayleigh;


    disp([num2str(i) ' setups out of ' num2str(nbrOfSetups)]);




end




 toc