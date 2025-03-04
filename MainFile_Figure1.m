%%=============================================================
%The file is used to generate the Figure 1 of the paper:
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
K_total = 10:5:30;


nbrOfRealizations = 1000;
nbrOfSetups = 50;
tau_p = 1;
tau_c = 200;


%Uplink transmit power per UE (W)
p = 0.2; %200 mW
%Create the power vector for all UEs (The uplink power is the same
%(p)at each UE)

SE_Distributed_OBE_MMSE_Monte_Sum = zeros(nbrOfSetups,length(K_total));
SE_Distributed_OBE_LS_Monte_Sum = zeros(nbrOfSetups,length(K_total));
SE_Distributed_OBE_MMSE_Analytical_Sum = zeros(nbrOfSetups,length(K_total));
SE_Distributed_OBE_LS_Analytical_Sum = zeros(nbrOfSetups,length(K_total));

SE_LSFD_RZF_MMSE_Monte_Sum = zeros(nbrOfSetups,length(K_total));
SE_LSFD_RZF_LS_Monte_Sum = zeros(nbrOfSetups,length(K_total));
SE_Distributed_RZF_MMSE_Monte_Sum = zeros(nbrOfSetups,length(K_total));
SE_Distributed_RZF_LS_Monte_Sum = zeros(nbrOfSetups,length(K_total));

SE_LSFD_LMMSE_MMSE_Monte_Sum = zeros(nbrOfSetups,length(K_total));
SE_LSFD_LMMSE_LS_Monte_Sum = zeros(nbrOfSetups,length(K_total));
SE_Distributed_LMMSE_MMSE_Monte_Sum = zeros(nbrOfSetups,length(K_total));
SE_Distributed_LMMSE_LS_Monte_Sum = zeros(nbrOfSetups,length(K_total));


% % % % % % % ----Parallel computing
% core_number = 2;     
% parpool('local',core_number);
% % % % Starting parallel pool (parpool) using the 'local' profile ...


for k = 1:length(K_total)

    K = K_total(k);
    pv = p*ones(1,K);

    % Rician fading channel
    probLoS = ones(M,1);


    for i = 1:nbrOfSetups

        [R_AP,H_LoS_Single_real,channelGain,channelGain_LoS,channelGain_NLoS] = functionGenerateSetupDeploy(M,K,N,1,1,probLoS);
        [H,H_LoS] = functionChannelGeneration(R_AP,H_LoS_Single_real,M,K,N,nbrOfRealizations);



        W_OBE_matrix_k_Initial = zeros(N,N,M);
        A_CE_matrix_k_Initial = zeros(N,N,M);
        A_CE_matrix_LS = zeros(N,N,M,K);


        for m = 1:M

            W_OBE_matrix_k_Initial(:,:,m) = eye(N);
            A_CE_matrix_k_Initial(:,:,m) = 1/(sqrt(p)*tau_p)*eye(N);

            for kk = 1:K

                A_CE_matrix_LS(:,:,m,kk) = 1/(sqrt(pv(kk))*tau_p)*eye(N);

            end

        end




        A_singleLayer = reshape(repmat(eye(M),1,K),M,M,K);
        [Pset] = functionPilotAllocation(R_AP,H_LoS_Single_real,A_singleLayer,M,K,N,tau_p,pv);


        [AMMSE] = functionMMSEChannelEstimator(R_AP,pv,M,K,N,tau_p,Pset);

        [Phi_MMSE,B_total_MMSE,C_total_MMSE] = functionMatrixGeneration(AMMSE,R_AP,pv,M,K,N,tau_p,Pset);
        [Phi_LS,~,C_total_LS] = functionMatrixGeneration(A_CE_matrix_LS,R_AP,pv,M,K,N,tau_p,Pset);


        [Hhat_MMSE] = functionChannelEstimates(AMMSE,H_LoS,H,nbrOfRealizations,M,K,N,tau_p,pv,Pset);
        [Hhat_LS] = functionChannelEstimates(A_CE_matrix_LS,H_LoS,H,nbrOfRealizations,M,K,N,tau_p,pv,Pset);


        %---Distributed

        %--OBE
        [V_OBE_Combining_Distributed,W_OBE_matrix_Distributed] = functionOBE_Combining_Distributed(H_LoS,Hhat_MMSE,AMMSE,Phi_MMSE,R_AP,Pset,M,N,K,pv,tau_p,nbrOfRealizations);
        [V_OBE_Combining_Distributed_LS,W_OBE_matrix_Distributed_LS] = functionOBE_Combining_Distributed(H_LoS,Hhat_LS,A_CE_matrix_LS,Phi_LS,R_AP,Pset,M,N,K,pv,tau_p,nbrOfRealizations);


        [SE_OBE_Monte_LSFD_MMSE,SE_OBE_Monte_MMSE] = functionComputeSE_Distributed_Monte(H,V_OBE_Combining_Distributed,tau_c,tau_p,nbrOfRealizations,N,K,M,pv);
        [SE_OBE_Monte_LSFD_LS,SE_OBE_Monte_LS] = functionComputeSE_Distributed_Monte(H,V_OBE_Combining_Distributed_LS,tau_c,tau_p,nbrOfRealizations,N,K,M,pv);

        [SE_OBE_Distributed_Analytical_MMSE] = functionComputeSE_Distributed_Analytical(AMMSE,W_OBE_matrix_Distributed,H_LoS,R_AP,Phi_MMSE,tau_c,tau_p,Pset,N,K,M,pv);
        [SE_OBE_Distributed_Analytical_LS] = functionComputeSE_Distributed_Analytical(A_CE_matrix_LS,W_OBE_matrix_Distributed_LS,H_LoS,R_AP,Phi_LS,tau_c,tau_p,Pset,N,K,M,pv);

        %--RZF
        [V_RZF_Combining_Distributed_LS] = functionZF_Combining_Distributed(Hhat_LS,nbrOfRealizations,M,N,K,pv);
        [SE_LSFD_RZF_Combining_LS,SE_Distributed_RZF_Combining_LS] = functionComputeSE_Distributed_Monte(H,V_RZF_Combining_Distributed_LS,tau_c,tau_p,nbrOfRealizations,N,K,M,pv);

        [V_RZF_Combining_Distributed_MMSE] = functionZF_Combining_Distributed(Hhat_MMSE,nbrOfRealizations,M,N,K,pv);
        [SE_LSFD_RZF_Combining_MMSE,SE_Distributed_RZF_Combining_MMSE] = functionComputeSE_Distributed_Monte(H,V_RZF_Combining_Distributed_MMSE,tau_c,tau_p,nbrOfRealizations,N,K,M,pv);



        %--LMMSE

        [V_LMMSE_Combining_Distributed_MMSE] = functionMMSE_Combining_Distributed(Hhat_MMSE,B_total_MMSE,C_total_MMSE,nbrOfRealizations,M,N,K,pv);
        [SE_LSFD_LMMSE_MMSE,SE_Distributed_LMMSE_Combining_MMSE] = functionComputeSE_Distributed_Monte(H,V_LMMSE_Combining_Distributed_MMSE,tau_c,tau_p,nbrOfRealizations,N,K,M,pv);

        B_total_LS = zeros(N,N,M);
        [V_LMMSE_Combining_Distributed_LS] = functionMMSE_Combining_Distributed(Hhat_LS,B_total_LS,C_total_LS,nbrOfRealizations,M,N,K,pv);
        [SE_LSFD_LMMSE_LS,SE_Distributed_LMMSE_Combining_LS] = functionComputeSE_Distributed_Monte(H,V_LMMSE_Combining_Distributed_LS,tau_c,tau_p,nbrOfRealizations,N,K,M,pv);



        SE_Distributed_OBE_MMSE_Monte_Sum(i,k) = sum(SE_OBE_Monte_MMSE);
        SE_Distributed_OBE_LS_Monte_Sum(i,k) = sum(SE_OBE_Monte_LS);

        SE_Distributed_OBE_MMSE_Analytical_Sum(i,k) = sum(SE_OBE_Distributed_Analytical_MMSE);
        SE_Distributed_OBE_LS_Analytical_Sum(i,k) = sum(SE_OBE_Distributed_Analytical_LS);

        SE_LSFD_RZF_MMSE_Monte_Sum(i,k) = sum(SE_LSFD_RZF_Combining_MMSE);
        SE_LSFD_RZF_LS_Monte_Sum(i,k) = sum(SE_LSFD_RZF_Combining_LS);

        SE_Distributed_RZF_MMSE_Monte_Sum(i,k) = sum(SE_Distributed_RZF_Combining_MMSE);
        SE_Distributed_RZF_LS_Monte_Sum(i,k) = sum(SE_Distributed_RZF_Combining_LS);

        SE_LSFD_LMMSE_MMSE_Monte_Sum(i,k) = sum(SE_LSFD_LMMSE_MMSE);
        SE_LSFD_LMMSE_LS_Monte_Sum(i,k) = sum(SE_LSFD_LMMSE_LS);

        SE_Distributed_LMMSE_MMSE_Monte_Sum(i,k) = sum(SE_Distributed_LMMSE_Combining_MMSE);
        SE_Distributed_LMMSE_LS_Monte_Sum(i,k) = sum(SE_Distributed_LMMSE_Combining_LS);



        disp([num2str(k) '-th K of ' num2str(length(K_total))]);
        disp([num2str(i) ' setups out of ' num2str(nbrOfSetups)]);



    end

end

toc