function [V_MMSE_Combining] = functionMMSE_Combining_Distributed(Hhat,B_total,C_total,nbrOfRealizations,M,N,K,p)
%%=============================================================
%The file is used to generate MMSE combining vectors in the paper:
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

%If only one transmit power is provided, use the same for all the UEs
if length(p) == 1
   p = p*ones(K,1);
end

%If no specific Level 1 transmit powers are provided, use the same as for
%the other levels
if nargin<12
    p1 = p;
end


%Store identity matrices of different sizes
eyeN = eye(N);





%Diagonal matrix with transmit powers and its square root
Dp = diag(p);

V_MMSE_Combining = zeros(M*N,K,nbrOfRealizations);




%% Go through all channel realizations
for n = 1:nbrOfRealizations


    %Go through all APs
    for m = 1:M


        %Extract channel estimate realizations from all UEs to AP l
        Hhatallj = reshape(Hhat(1+(m-1)*N:m*N,n,:),[N K]);
        
   
        %Compute MMSE combining

        V_MMSE_Combining((m-1)*N+1:m*N,:,n) = ((Hhatallj*Dp*Hhatallj') + B_total(:,:,m) + B_total(:,:,m)' + C_total(:,:,m) + eyeN)\(Hhatallj*Dp);

    end
end
        



V_MMSE_Combining = permute(V_MMSE_Combining,[1 3 2]);