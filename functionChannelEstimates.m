function [Hhat] = functionChannelEstimates(RPsi,HMean,H,nbrOfRealizations,M,K,N,tau_p,p,Pset)

%%=============================================================
%The file is used to generate the channel estimate based on arbitrary channel estimator of the paper:
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

%Prepare to store MMSE channel estimates
Hhat = zeros(M*N,nbrOfRealizations,K);

%Generate realizations of normalized noise 
Np = sqrt(0.5)*(randn(N,nbrOfRealizations,M,K) + 1i*randn(N,nbrOfRealizations,M,K));


for m = 1:M
    for k = 1:K
        
        yp = zeros(N,nbrOfRealizations);
        yMean = zeros(N,nbrOfRealizations);
        inds = Pset(:,k); 
        
        for z = 1:length(inds)
            
            yp = yp + sqrt(p(inds(z)))*tau_p*H((m-1)*N+1:m*N,:,inds(z));
            yMean = yMean + sqrt(p(inds(z)))*tau_p*HMean((m-1)*N+1:m*N,:,inds(z));
            
        end
        
        yp = yp + sqrt(tau_p)*Np(:,:,m,k);
        
      
        for z = 1:length(inds)
            
            Hhat((m-1)*N+1:m*N,:,inds(z)) = HMean((m-1)*N+1:m*N,:,inds(z)) + RPsi(:,:,m,inds(z))*(yp-yMean);
            
        end
    end
end

