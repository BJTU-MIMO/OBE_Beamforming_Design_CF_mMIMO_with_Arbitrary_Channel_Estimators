function [V_Precoding_Normalized] = functionDownlink_Precoding_Design(channelGain,V_Precoding,nbrOfRealizations,N,K,M,p)
%%=============================================================
%The file is used to generate downlink precoding vectors in the paper:
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

%The power is allocated proportional to the channel quality
Dx = [];
for s = 1:M

    temp = channelGain(s,:)./sum(channelGain(s,:));
    Dx = [Dx;temp];

end


V_Precoding_Normalized =zeros(M*N,nbrOfRealizations,K);
Normalized_Factors = zeros(M,K);
Power_Factors = zeros(M,K);


for n = 1:nbrOfRealizations


    for m = 1:M
        
        V = reshape(V_Precoding(1+(m-1)*N:m*N,n,:),[N K]);

        for k = 1:K

            v = V(:,k);

            Normalized_Factors(m,k) = Normalized_Factors(m,k) + norm(v)^2/nbrOfRealizations;

        end
    end
end


for n = 1:nbrOfRealizations

    for m = 1:M

        for k = 1:K

            V_Precoding_Normalized(1+(m-1)*N:m*N,n,k) = sqrt(Dx(m,k)*p(k)/Normalized_Factors(m,k))*V_Precoding(1+(m-1)*N:m*N,n,k);
            Power_Factors(m,k) = sqrt(Dx(m,k)*p(k)/Normalized_Factors(m,k));

        end
    end
end




