function [Pset] = functionPilotAllocation( R_AP,HMean_Withoutphase,A_singleLayer,M,K,N,tau_p,p)
%%=============================================================
%The file is used to allocate the pilot of the paper:
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

%Pilot set initialize
Pset=1:tau_p;


for z=1:(K/tau_p)-1
    Pset=[Pset;((tau_p*z)+1)*ones(1,tau_p)];
    ind=[];
    for s=1:tau_p
        %Check fot the coherent interference levels
        [coherentx,~] = functionMMSE_interferenceLevels( R_AP,HMean_Withoutphase,A_singleLayer,M,tau_p,N,tau_p,p,Pset);
        %Select the UE index that creates least interference
        if s ~=1
            coherentx(ind)=nan;
        end
        [~,ind(s)]=min(coherentx);
        x=1:tau_p;
        x(ind)=[];
        Pset(z+1,x)=(z*tau_p)+s+1;
        
    end
    
end
%Order the pilot allocation set
for i=1:K
    [~,c]=find(Pset==i);
    temp=Pset(:,c);
    temp(temp==i)=[];
    PsetOrdered(:,i)=[i;temp];
end

%The output file
Pset=PsetOrdered;

end

