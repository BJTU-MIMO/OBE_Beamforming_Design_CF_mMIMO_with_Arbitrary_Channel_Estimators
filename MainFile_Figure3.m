%%=============================================================
%The file is used to generate the Figure 3 of the paper:
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


M = 20;
N_total = 1:1:6;
K = 10;


nbrOfRealizations = 1000;
nbrOfSetups = 80;

tau_p = 1;
tau_c = 200;

%Uplink transmit power per UE (W)
p = 0.2; %200 mW
%Create the power vector for all UEs (The uplink power is the same
%(p)at each UE)
pv = p*ones(1,K);

SE_OBE_Downlink_Monte_MMSE_total = zeros(K,nbrOfSetups,length(N_total));
SE_OBE_Downlink_Analytical_MMSE_total = zeros(K,nbrOfSetups,length(N_total));
SE_OBE_Downlink_Monte_LS_total = zeros(K,nbrOfSetups,length(N_total));
SE_OBE_Downlink_Analytical_LS_total = zeros(K,nbrOfSetups,length(N_total));

SE_RZF_Downlink_Monte_MMSE_total = zeros(K,nbrOfSetups,length(N_total));
SE_RZF_Downlink_Monte_LS_total = zeros(K,nbrOfSetups,length(N_total));

SE_LMMSE_Downlink_Monte_MMSE_total = zeros(K,nbrOfSetups,length(N_total));
SE_LMMSE_Downlink_Monte_LS_total = zeros(K,nbrOfSetups,length(N_total));

% % % % % % % ----Parallel computing
core_number = 2; 
parpool('local',core_number);
% % % % Starting parallel pool (parpool) using the 'local' profile ...


for n = 1:length(N_total)

    N = N_total(n);

    parfor i = 1:nbrOfSetups

        probLoS = ones(M,1);

        [R_AP,H_LoS_Single_real,channelGain,channelGain_LoS,channelGain_NLoS] = functionGenerateSetupDeploy(M,K,N,1,1,probLoS);
        [H,H_LoS] = functionChannelGeneration(R_AP,H_LoS_Single_real,M,K,N,nbrOfRealizations);

        A_CE_matrix_LS = zeros(N,N,M,K);

        for m = 1:M
            for k = 1:K

                A_CE_matrix_LS(:,:,m,k) = 1/(sqrt(pv(k))*tau_p)*eye(N);

            end
        end

        A_singleLayer = reshape(repmat(eye(M),1,K),M,M,K);


        [Pset] = functionPilotAllocation(R_AP,H_LoS_Single_real,A_singleLayer,M,K,N,tau_p,pv);


        [AMMSE] = functionMMSEChannelEstimator(R_AP,pv,M,K,N,tau_p,Pset);

        [Phi_MMSE,B_total_MMSE,C_total_MMSE] = functionMatrixGeneration(AMMSE,R_AP,pv,M,K,N,tau_p,Pset);
        [Phi_LS,~,C_total_LS] = functionMatrixGeneration(A_CE_matrix_LS,R_AP,pv,M,K,N,tau_p,Pset);



        [Hhat_MMSE] = functionChannelEstimates(AMMSE,H_LoS,H,nbrOfRealizations,M,K,N,tau_p,pv,Pset);
        [Hhat_LS] = functionChannelEstimates(A_CE_matrix_LS,H_LoS,H,nbrOfRealizations,M,K,N,tau_p,pv,Pset);


        %--OBE
        [V_OBE_Combining_Distributed_MMSE,W_OBE_matrix_MMSE] = functionOBE_Combining_Distributed(H_LoS,Hhat_MMSE,AMMSE,Phi_MMSE,R_AP,Pset,M,N,K,pv,tau_p,nbrOfRealizations);
        [V_OBE_Precoding_Normalized_MMSE] = functionDownlink_Precoding_Design(channelGain,V_OBE_Combining_Distributed_MMSE,nbrOfRealizations,N,K,M,pv);

        [SE_OBE_Downlink_Monte_MMSE] = functionComputeSE_Distributed_Downlink_Monte(H,V_OBE_Precoding_Normalized_MMSE,tau_c,tau_p,nbrOfRealizations,N,K,M);
        [SE_OBE_Downlink_Analytical_MMSE] = functionComputeSE_Distributed_Downlink_Analytical(AMMSE,W_OBE_matrix_MMSE,H_LoS,channelGain,R_AP,Phi_MMSE,tau_c,tau_p,Pset,N,K,M,pv);


        [V_OBE_Combining_Distributed_LS,W_OBE_matrix_LS] = functionOBE_Combining_Distributed(H_LoS,Hhat_LS,A_CE_matrix_LS,Phi_LS,R_AP,Pset,M,N,K,pv,tau_p,nbrOfRealizations);
        [V_OBE_Precoding_Normalized_LS] = functionDownlink_Precoding_Design(channelGain,V_OBE_Combining_Distributed_LS,nbrOfRealizations,N,K,M,pv);

        [SE_OBE_Downlink_Monte_LS] = functionComputeSE_Distributed_Downlink_Monte(H,V_OBE_Precoding_Normalized_LS,tau_c,tau_p,nbrOfRealizations,N,K,M);
        [SE_OBE_Downlink_Analytical_LS] = functionComputeSE_Distributed_Downlink_Analytical(A_CE_matrix_LS,W_OBE_matrix_LS,H_LoS,channelGain,R_AP,Phi_LS,tau_c,tau_p,Pset,N,K,M,pv);

        %--RZF
        [V_RZF_Combining_Distributed_MMSE] = functionZF_Combining_Distributed(Hhat_MMSE,nbrOfRealizations,M,N,K,pv);
        [V_RZF_Combining_Distributed_Normalized_MMSE] = functionDownlink_Precoding_Design(channelGain,V_RZF_Combining_Distributed_MMSE,nbrOfRealizations,N,K,M,pv);
        [SE_RZF_Downlink_Monte_MMSE] = functionComputeSE_Distributed_Downlink_Monte(H,V_RZF_Combining_Distributed_Normalized_MMSE,tau_c,tau_p,nbrOfRealizations,N,K,M);


        [V_RZF_Combining_Distributed_RZF_LS] = functionZF_Combining_Distributed(Hhat_LS,nbrOfRealizations,M,N,K,pv);
        [V_RZF_Combining_Distributed_Normalized_LS] = functionDownlink_Precoding_Design(channelGain,V_RZF_Combining_Distributed_RZF_LS,nbrOfRealizations,N,K,M,pv);
        [SE_RZF_Downlink_Monte_LS] = functionComputeSE_Distributed_Downlink_Monte(H,V_RZF_Combining_Distributed_Normalized_LS,tau_c,tau_p,nbrOfRealizations,N,K,M);

        %--LMMSE

        [V_LMMSE_Combining_Distributed_MMSE] = functionMMSE_Combining_Distributed(Hhat_MMSE,B_total_MMSE,C_total_MMSE,nbrOfRealizations,M,N,K,pv);
        [V_MMSE_Combining_Distributed_Normalized_MMSE] = functionDownlink_Precoding_Design(channelGain,V_LMMSE_Combining_Distributed_MMSE,nbrOfRealizations,N,K,M,pv);
        [SE_MMSE_Downlink_Monte_MMSE] = functionComputeSE_Distributed_Downlink_Monte(H,V_MMSE_Combining_Distributed_Normalized_MMSE,tau_c,tau_p,nbrOfRealizations,N,K,M);

        B_total_LS = zeros(N,N,M);
        [V_LMMSE_Combining_Distributed_LS] = functionMMSE_Combining_Distributed(Hhat_LS,B_total_LS,C_total_LS,nbrOfRealizations,M,N,K,pv);
        [V_MMSE_Combining_Distributed_Normalized_LS] = functionDownlink_Precoding_Design(channelGain,V_LMMSE_Combining_Distributed_LS,nbrOfRealizations,N,K,M,pv);
        [SE_MMSE_Downlink_Monte_LS] = functionComputeSE_Distributed_Downlink_Monte(H,V_MMSE_Combining_Distributed_Normalized_LS,tau_c,tau_p,nbrOfRealizations,N,K,M);



        SE_OBE_Downlink_Monte_MMSE_total(:,i,n) = SE_OBE_Downlink_Monte_MMSE;
        SE_OBE_Downlink_Analytical_MMSE_total(:,i,n) = SE_OBE_Downlink_Analytical_MMSE;
        SE_OBE_Downlink_Monte_LS_total(:,i,n) = SE_OBE_Downlink_Monte_LS;
        SE_OBE_Downlink_Analytical_LS_total(:,i,n) = SE_OBE_Downlink_Analytical_LS;

        SE_RZF_Downlink_Monte_MMSE_total(:,i,n) = SE_RZF_Downlink_Monte_MMSE;
        SE_RZF_Downlink_Monte_LS_total(:,i,n) = SE_RZF_Downlink_Monte_LS;

        SE_LMMSE_Downlink_Monte_MMSE_total(:,i,n) = SE_MMSE_Downlink_Monte_MMSE;
        SE_LMMSE_Downlink_Monte_LS_total(:,i,n) = SE_MMSE_Downlink_Monte_LS;

        disp([num2str(n) '-th N of ' num2str(length(N_total))]);
        disp([num2str(i) ' setups out of ' num2str(nbrOfSetups)]);


    end

end
toc

