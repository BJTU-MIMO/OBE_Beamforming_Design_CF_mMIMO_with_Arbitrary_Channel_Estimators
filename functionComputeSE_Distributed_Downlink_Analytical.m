function [SE] = functionComputeSE_Distributed_Downlink_Analytical(A,W,G_LoS_eff,channelGain,R,Phi,tau_c,tau_p,Pset,N,K,M,p)
%%=============================================================
%The file is used to compute the downlink achievable SE by analytical results in the paper:
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

% %If only one transmit power is provided, use the same for all the UEs
if length(p) == 1
   p = p*ones(K,1);
end

%If no specific Level 1 transmit powers are provided, use the same as for
%the other levels
if nargin<12
    p1 = p;
end


%Store identity matrices of different sizes


%Compute the prelog factor
prelogFactor = (1-tau_p/tau_c);



%The power is allocated proportional to the channel quality
Dx = [];
for s = 1:M

    temp = channelGain(s,:)./sum(channelGain(s,:));
    Dx = [Dx;temp];
    
end


Power_Factors = zeros(M,K);


SE = zeros(K,1);
G_LoS_eff = reshape(G_LoS_eff(:,1,:),M*N,K);

% Define matrices
Gbar_mkl = zeros(N,N,M,K,K);
B_mlk = zeros(N,N,M,K,K);
Rbar_mk = zeros(N,N,M,K);
Phi_cross_mlk = zeros(N,N,M,K,K);

%% Compute the matrices applied in the OBE combining design

for k = 1:K
    for m = 1:M

        Power_Factors(m,k) = sqrt(Dx(m,k)*p(k)/(trace(W(:,:,m,k)'*W(:,:,m,k)*G_LoS_eff((m-1)*N+1:m*N,k)*G_LoS_eff((m-1)*N+1:m*N,k)') + trace(W(:,:,m,k)'*W(:,:,m,k)*tau_p*A(:,:,m,k)*Phi(:,:,m,k)*A(:,:,m,k)')));

        for l = 1:K

            Gbar_mkk = G_LoS_eff((m-1)*N+1:m*N,k)*G_LoS_eff((m-1)*N+1:m*N,k)';

            Rbar_mk(:,:,m,k) = Gbar_mkk + sqrt(p(k))*tau_p*R(:,:,m,k)*A(:,:,m,k)';


            Gbar_mkl(:,:,m,k,l) = G_LoS_eff((m-1)*N+1:m*N,k)*G_LoS_eff((m-1)*N+1:m*N,l)';

            
            B_mlk(:,:,m,l,k) = G_LoS_eff((m-1)*N+1:m*N,l)*G_LoS_eff((m-1)*N+1:m*N,k)' + sqrt(p(l))*tau_p*A(:,:,m,l)*R(:,:,m,k);


            Phi_cross_mlk(:,:,m,l,k) = A(:,:,m,l)*R(:,:,m,k);


        end

    end
end




numerator = zeros(K,1);
mu_mkl = zeros(K,1);
gamma = zeros(K,K,M);
denominator3 = zeros(K,1);


for k = 1:K
    for m = 1:M


        numerator(k) = numerator(k) + Power_Factors(m,k)*trace(W(:,:,m,k)'*Rbar_mk(:,:,m,k));


        denominator3(k) = denominator3(k) + trace(W(:,:,m,k)'*W(:,:,m,k)*Gbar_mkl(:,:,m,k,k)) + trace(W(:,:,m,k)'*W(:,:,m,k)*tau_p*A(:,:,m,k)*Phi(:,:,m,k)*A(:,:,m,k)');


        for l = 1:K

            mu_mkl(k) = mu_mkl(k) + Power_Factors(m,l)^2*trace(W(:,:,m,l)'*Gbar_mkl(:,:,m,k,k)*W(:,:,m,l)*Gbar_mkl(:,:,m,l,l)) + Power_Factors(m,l)^2*trace(W(:,:,m,l)'*R(:,:,m,k)*W(:,:,m,l)*Gbar_mkl(:,:,m,l,l))...
                + Power_Factors(m,l)^2*trace(W(:,:,m,l)'*Gbar_mkl(:,:,m,k,k)*W(:,:,m,l)*tau_p*A(:,:,m,l)*Phi(:,:,m,l)*A(:,:,m,l)') + Power_Factors(m,l)^2*trace(W(:,:,m,l)'*R(:,:,m,k)*W(:,:,m,l)*tau_p*A(:,:,m,l)*Phi(:,:,m,l)*A(:,:,m,l)');
            
            gamma(k,l,m) = Power_Factors(m,l)*trace(W(:,:,m,l)*Gbar_mkl(:,:,m,l,k));



            if any(l == Pset(:,k))

                mu_mkl(k) = mu_mkl(k) + Power_Factors(m,l)^2*sqrt(p(k))*tau_p*trace(W(:,:,m,l)'*Gbar_mkl(:,:,m,k,l))*trace(W(:,:,m,l)*Phi_cross_mlk(:,:,m,l,k)) + Power_Factors(m,l)^2*sqrt(p(k))*tau_p*trace(W(:,:,m,l)'*Phi_cross_mlk(:,:,m,l,k)')*trace(W(:,:,m,l)*Gbar_mkl(:,:,m,l,k))...
                    + Power_Factors(m,l)^2*p(k)*tau_p^2*abs(trace(W(:,:,m,l)'*Phi_cross_mlk(:,:,m,l,k)'))^2;


                gamma(k,l,m) = Power_Factors(m,l)*trace(W(:,:,m,l)*B_mlk(:,:,m,l,k));


            end

        end
    end
end


denominator2_1 = 0;
denominator2_2 = 0;
denominator2 = zeros(K,1);

for k = 1:K

    for l = 1:K

        for m = 1:M

            denominator2_1 = denominator2_1 + gamma(k,l,m);
            denominator2_2 = denominator2_2 + abs(gamma(k,l,m))^2;

        end

        denominator2_1 = abs(denominator2_1)^2;

        denominator2(k) = denominator2(k) + denominator2_1 - denominator2_2;

        denominator2_1 = 0;
        denominator2_2 = 0;

    end
end


for k = 1:K


    Numerator =  numerator(k)*numerator(k)';

    Denominator = mu_mkl(k) - numerator(k)*numerator(k)' + denominator2(k) + 1;

    SE(k) = prelogFactor*real(log2(1+ Numerator/Denominator));

end

