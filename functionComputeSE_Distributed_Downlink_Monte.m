function [SE_Downlink] = functionComputeSE_Distributed_Downlink_Monte(H,V_Precoding,tau_c,tau_p,nbrOfRealizations,N,K,M)
%%=============================================================
%The file is used to compute the downlink achievable SE by Monte-Carlo simulations in the paper:
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

%Compute the prelog factor
prelogFactor = (1-tau_p/tau_c);

%Prepare to store simulation results
SE_Downlink = zeros(K,1);

%Compute sum of all estimation error correlation matrices at every BS




%Prepare to save simulation results

Term1 = zeros(K,M);
Term2 = zeros(K,K);

%% Go through all channel realizations
for n = 1:nbrOfRealizations

    Hallj_all_APs = reshape(H(:,n,:),[M*N K]);

    V_all_APs = reshape(V_Precoding(:,n,:),[M*N K]);



    %Go through all APs
    for m = 1:M

        %Extract channel realizations from all UEs to AP l
        Hallj = reshape(H(1+(m-1)*N:m*N,n,:),[N K]);

        V = reshape(V_Precoding(1+(m-1)*N:m*N,n,:),[N K]);


        for k = 1:K

            v = V(:,k);


            Term1(k,m) = Term1(k,m) + v'*Hallj(:,k)/nbrOfRealizations;

            for l = 1:K

                Term2(k,l) = Term2(k,l) + abs(V_all_APs(:,l)'*Hallj_all_APs(:,k))^2/(nbrOfRealizations*M); % Repeating iterations of M

            end

        end
    end
end



Term1_K = abs(sum(Term1,2)).^2;
Term2_K = sum(Term2,2);


for k = 1:K

    SE_Downlink(k) = prelogFactor*log2(1 + Term1_K(k)/(Term2_K(k) - Term1_K(k) + 1));

end
