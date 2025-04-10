function [V_OBE_Combining_Distributed,W_OBE_matrix] = functionOBE_Combining_Distributed(G_LoS_eff,Ghat,A,Phi,R,Pset,M,N,K,p,tau_p,nbrOfRealizations)
%%=============================================================
%The file is used to generate the distributed OBE combining vectors of the paper:
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


G_LoS_eff = reshape(G_LoS_eff(:,1,:),M*N,K);


% Define matrices
Gbar_mkl = zeros(N,N,M,K,K);
Bbar_lk_blk = zeros(M*N^2,M*N^2,K,K);
Bhat_lk = zeros(M*N^2,M*N^2,K,K);
B_mlk = zeros(N,N,M,K,K);
Rbar_mk = zeros(N,N,M,K);
Phi_cross_mlk = zeros(N,N,M,K,K);
Gbar_lk_2 = zeros(M*N^2,M*N^2,K,K);
Gbarbar_lk_blk = zeros(M*N^2,M*N^2,K,K);

rbar_k = zeros(M*N^2,K);


Lambda_kl_1 = zeros(M*N^2,M*N^2,K,K);
Lambda_kl_2 = zeros(M*N^2,M*N^2,K,K);
Lambda_kl_3 = zeros(M*N^2,M*N^2,K,K);
Lambda_kl_4 = zeros(M*N^2,M*N^2,K,K);
Lambda_k_5 = zeros(M*N^2,M*N^2,K);

W_OBE_matrix = zeros(N,N,M,K);

V_OBE_Combining_Distributed = zeros(M*N,nbrOfRealizations,K);


%% Compute the matrices applied in the OBE combining design

for k = 1:K
    for m = 1:M
        for l = 1:K

            Gbar_mkk = G_LoS_eff((m-1)*N+1:m*N,k)*G_LoS_eff((m-1)*N+1:m*N,k)';

            Rbar_mk(:,:,m,k) = Gbar_mkk + sqrt(p(k))*tau_p*R(:,:,m,k)*A(:,:,m,k)';
         

            Lambda_k_5((m-1)*N^2+1:m*N^2,(m-1)*N^2+1:m*N^2,k) = kron(Gbar_mkk.',eye(N)) + kron((tau_p*A(:,:,m,k)*Phi(:,:,m,k)*A(:,:,m,k)').',eye(N));

            Gbar_mkl(:,:,m,k,l) = G_LoS_eff((m-1)*N+1:m*N,k)*G_LoS_eff((m-1)*N+1:m*N,l)';


            B_mlk(:,:,m,l,k) = G_LoS_eff((m-1)*N+1:m*N,l)*G_LoS_eff((m-1)*N+1:m*N,k)' + sqrt(p(l))*tau_p*R(:,:,m,l)*A(:,:,m,k)';
            b_mlk_vec = B_mlk(:,:,m,l,k);
            b_mlk_vec = b_mlk_vec(:);

            Bbar_lk_blk((m-1)*N^2+1:m*N^2,(m-1)*N^2+1:m*N^2,l,k) = b_mlk_vec*b_mlk_vec';

            Phi_cross_mlk(:,:,m,l,k) = R(:,:,m,l)*A(:,:,m,k)';

            gbar_mlk = G_LoS_eff((m-1)*N+1:m*N,l)*G_LoS_eff((m-1)*N+1:m*N,k)';

            gbar_mlk = gbar_mlk(:);
            Gbarbar_lk_blk((m-1)*N^2+1:m*N^2,(m-1)*N^2+1:m*N^2,l,k) = gbar_mlk*gbar_mlk';

    


        end

        rbar_mk = Rbar_mk(:,:,m,k);
        rbar_mk = rbar_mk(:);
        rbar_k((m-1)*N^2+1:m*N^2,k) = rbar_mk;

    end
end




for k = 1:K
    for l = 1:K

        for m = 1:M

            for mm = 1:M


                gbar_mlk = Gbar_mkl(:,:,m,l,k);
                gbar_mlk = gbar_mlk(:);

                gbar_mmlk = Gbar_mkl(:,:,mm,l,k);
                gbar_mmlk = gbar_mmlk(:);



                Gbar_lk_2((m-1)*N^2+1:m*N^2,(mm-1)*N^2+1:mm*N^2,l,k) = gbar_mlk*gbar_mmlk';


                b_mlk = B_mlk(:,:,m,l,k);
                b_mlk = b_mlk(:);

                b_mmlk = B_mlk(:,:,mm,l,k);
                b_mmlk = b_mmlk(:);

                Bhat_lk((m-1)*N^2+1:m*N^2,(mm-1)*N^2+1:mm*N^2,l,k) = b_mlk*b_mmlk';

            end


            Lambda_kl_1((m-1)*N^2+1:m*N^2,(m-1)*N^2+1:m*N^2,k,l) = p(l)*kron(Gbar_mkl(:,:,m,k,k).',Gbar_mkl(:,:,m,l,l)) + p(l)*kron(Gbar_mkl(:,:,m,k,k).',R(:,:,m,l))...
                +p(l)*tau_p*kron((A(:,:,m,k)*Phi(:,:,m,k)*A(:,:,m,k)').',R(:,:,m,l)) + p(l)*tau_p*kron((A(:,:,m,k)*Phi(:,:,m,k)*A(:,:,m,k)').',Gbar_mkl(:,:,m,l,l));


            rwave_mlk = Phi_cross_mlk(:,:,m,l,k);

            rwave_mlk = rwave_mlk(:);

            Lambda_kl_2((m-1)*N^2+1:m*N^2,(m-1)*N^2+1:m*N^2,k,l) = p(l)*(sqrt(p(l))*tau_p*gbar_mlk*rwave_mlk' + sqrt(p(l))*tau_p*rwave_mlk*gbar_mlk' + p(l)*tau_p^2*(rwave_mlk*rwave_mlk'));


            
        end


        Lambda_kl_3(:,:,k,l) = p(l)*Gbar_lk_2(:,:,l,k) - p(l)*Gbarbar_lk_blk(:,:,l,k);

        Lambda_kl_4(:,:,k,l) = p(l)*Bhat_lk(:,:,l,k) - p(l)*Bbar_lk_blk(:,:,l,k) - p(l)*Gbar_lk_2(:,:,l,k) + p(l)*Gbarbar_lk_blk(:,:,l,k);



    end
end




%% Distributed OBE combining design

for k = 1:K

    Lambda = zeros(M*N^2,M*N^2);

    for l = 1:K

        Lambda = Lambda + Lambda_kl_1(:,:,k,l) + Lambda_kl_3(:,:,k,l);

        if any(l == Pset(:,k))

            Lambda = Lambda + Lambda_kl_2(:,:,k,l) + Lambda_kl_4(:,:,k,l);

        end

    end
    
    Lambda = Lambda - p(k)*(rbar_k(:,k)*rbar_k(:,k)') + Lambda_k_5(:,:,k);
    W_OBE_vector = Lambda\rbar_k(:,k);
        

    for m = 1:M

        W_OBE_vector_m = W_OBE_vector((m-1)*N^2+1:m*N^2);

        W_OBE_matrix_m = reshape(W_OBE_vector_m,[N,N]);

        W_OBE_matrix(:,:,m,k) = W_OBE_matrix_m;

        V_OBE_Combining_Distributed((m-1)*N+1:m*N,:,k) = W_OBE_matrix(:,:,m,k)*Ghat((m-1)*N+1:m*N,:,k);

    end

    clear Gamma W_OBE_vector W_OBE_matrix_m 

end

clear Gbar_mkl Bbar_lk_blk Bhat_lk B_mlk Rbar_mk Phi_cross_mlk Gbar_lk_2 Gbarbar_lk_blk rbar_k
