function [AMMSE] = functionMMSEChannelEstimator(R,p,M,K,N,tau_p,Pset)
%%=============================================================
%The file is used to generate the MMSE estimator-based matrix of the paper:
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

% Prepare to store the result
AMMSE = zeros(N,N,M,K);


% Go through all sub-arrays          
for m = 1:M
    
    % Go through all UEs
    for k = 1:K
        
        % Compute the UEs indexes that use the same pilot as UE k
        inds = Pset(:,k);
        PsiInv = zeros(N,N);
        
        % Go through all UEs that use the same pilot as UE k 
        for z = 1:length(inds)   
            
            PsiInv = PsiInv + p(inds(z))*tau_p*R(:,:,m,inds(z));

        end
            PsiInv = PsiInv + eye(N);



            AMMSE(:,:,m,k) = sqrt(p(k))*R(:,:,m,k)/PsiInv;
            

            
    end
end

        
                        

            
